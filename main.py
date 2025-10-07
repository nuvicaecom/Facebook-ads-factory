import os
import itertools
import subprocess
import json
import re
import tempfile
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# ===== Config =====
SONG_BG_VOL = 0.1   # background music level (0.10–0.25 is typical)
FPS = 30

# Loudness targets suitable for Meta (FB/IG) ads
TARGET_I   = -14.0   # LUFS integrated loudness
TARGET_TP  = -1.5    # dB True Peak (headroom for transcode)
TARGET_LRA = 11.0    # Loudness range target

# Trim option: use only the first N seconds of the HOOK before merging
TRIM_HOOK_HEAD   = True     # <- toggle here
HOOK_HEAD_SECONDS = 4.0      # length of hook head to keep when TRIM_HOOK_HEAD=True

# Export options
EXPORT_NOSONG_VERSION = False     # <- always export a no-song version
EXPORT_SONG_VERSIONS  = True    # <- also export versions with background songs (if any)

# ===== CTA (end clip) =====
APPEND_CTA = True          # master toggle
CTAS_DIRNAME = 'ctas'      # folder containing CTA .mp4 clips

# ===== Directories =====
hooks_dir  = Path('hooks')
ads_dir    = Path('ads')
songs_dir  = Path('songs')
ctas_dir   = Path(CTAS_DIRNAME)
output_dir = Path('output')
temp_dir   = Path('temp')
output_dir.mkdir(parents=True, exist_ok=True)
temp_dir.mkdir(parents=True, exist_ok=True)   # ensure temp exists

# ===== GPU encoder detection / selection =====
def detect_best_encoder():
    try:
        out = subprocess.run(
            ['ffmpeg', '-hide_banner', '-encoders'],
            text=True, capture_output=True, check=True
        ).stdout.lower()
    except Exception:
        return 'libx264'
    for enc in ('h264_nvenc', 'h264_qsv', 'h264_amf', 'h264_videotoolbox'):
        if enc in out:
            return enc
    return 'libx264'

VIDEO_ENCODER = os.environ.get('VIDEO_ENCODER', 'auto')
if VIDEO_ENCODER == 'auto':
    VIDEO_ENCODER = detect_best_encoder()

def video_encoder_args():
    """Return encoder-specific args (includes GOP length)."""
    g = str(FPS * 2)
    if VIDEO_ENCODER == 'h264_nvenc':
        return ['-c:v', 'h264_nvenc', '-preset', 'p5', '-tune', 'hq', '-rc', 'vbr', '-cq', '19', '-bf', '2', '-g', g]
    if VIDEO_ENCODER == 'h264_qsv':
        return ['-c:v', 'h264_qsv', '-global_quality', '23', '-g', g]
    if VIDEO_ENCODER == 'h264_amf':
        return ['-c:v', 'h264_amf', '-quality', 'quality', '-g', g]
    if VIDEO_ENCODER == 'h264_videotoolbox':
        return ['-c:v', 'h264_videotoolbox', '-g', g]
    # Fallback CPU x264 with strict GOP behavior
    return [
        '-c:v', 'libx264',
        '-profile:v', 'high', '-level:v', '4.0',
        '-x264-params', f'scenecut=0:open_gop=0:min-keyint={g}:keyint={g}'
    ]

def video_common_compat():
    """Compatibility flags for social platforms (AVC High@L4.0, yuv420p, avc1 tag)."""
    return ['-pix_fmt', 'yuv420p', '-profile:v', 'high', '-level:v', '4.0', '-tag:v', 'avc1']

print(f"[encoder] Using {VIDEO_ENCODER}")

# ===== Utils =====
def run_ffmpeg(args, **kwargs):
    """Run ffmpeg with quieter logging for cleaner parallel output."""
    base = ['ffmpeg', '-hide_banner', '-loglevel', 'error']
    subprocess.run(base + args, check=True, **kwargs)

def probe_video_size(path: Path):
    """Return (width, height) using ffprobe."""
    result = subprocess.run(
        [
            'ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height',
            '-of', 'json', str(path)
        ],
        text=True, capture_output=True, check=True
    )
    info = json.loads(result.stdout or '{}')
    stream = info.get('streams', [{}])[0]
    return int(stream.get('width', 0)), int(stream.get('height', 0))

def has_audio(path: Path) -> bool:
    """Return True if file has at least one audio stream."""
    try:
        result = subprocess.run(
            [
                'ffprobe', '-v', 'error',
                '-select_streams', 'a',
                '-show_entries', 'stream=index',
                '-of', 'json', str(path)
            ],
            text=True, capture_output=True, check=True
        )
        info = json.loads(result.stdout or '{}')
        return bool(info.get('streams'))
    except subprocess.CalledProcessError:
        return False

def ensure_1080x1920(path: Path):
    """If video is not 1080x1920, resize (keep AR with pad) and replace in-place."""
    if path.suffix.lower() != '.mp4':
        return
    try:
        w, h = probe_video_size(path)
    except subprocess.CalledProcessError:
        print(f"[warn] ffprobe failed for {path}")
        return

    if (w, h) == (1080, 1920):
        return

    print(f"[resize] {path.name}: {w}x{h} -> 1080x1920")
    tmp_out = path.with_suffix('.tmp.mp4')
    vf = (
        "scale=1080:1920:force_original_aspect_ratio=decrease,"
        "pad=1080:1920:(ow-iw)/2:(oh-ih)/2"
    )
    run_ffmpeg([
        '-y', '-i', str(path),
        '-vf', vf,
        '-r', str(FPS),
        *video_encoder_args(),           # GPU or x264
        *video_common_compat(),          # yuv420p + profiles
        '-c:a', 'aac', '-b:a', '192k',
        '-movflags', '+faststart',
        str(tmp_out)
    ])
    os.replace(tmp_out, path)

def list_media(dirpath: Path, exts):
    if not dirpath.exists():
        return []
    return [p.name for p in dirpath.iterdir() if p.is_file() and p.suffix.lower() in exts]

# ===== Loudness Normalization (robust two-pass) =====
LN_BLOCK_RE = re.compile(r'\{[^{}]*\}', re.DOTALL)
EXPECTED_KEYS = {"input_i", "input_tp", "input_lra", "input_thresh", "target_offset"}

def loudnorm_measure(in_path: Path):
    """
    First pass: measure loudness; return dict with measured values, or None if not measurable.
    We DON'T use run_ffmpeg here so we can parse stderr.
    """
    proc = subprocess.run([
        'ffmpeg', '-hide_banner', '-nostats',
        '-i', str(in_path),
        '-af', f"loudnorm=I={TARGET_I}:TP={TARGET_TP}:LRA={TARGET_LRA}:print_format=json",
        '-f', 'null', '-'
    ], text=True, capture_output=True)

    stderr = proc.stderr or ""
    blocks = LN_BLOCK_RE.findall(stderr)

    data = None
    for b in blocks:
        try:
            j = json.loads(b)
        except Exception:
            continue
        if EXPECTED_KEYS.issubset(set(j.keys())):
            data = j
            break

    if not data:
        return None

    return {
        'measured_I': data.get('input_i'),
        'measured_TP': data.get('input_tp'),
        'measured_LRA': data.get('input_lra'),
        'measured_thresh': data.get('input_thresh'),
        'offset': data.get('target_offset')
    }

def loudnorm_apply(in_path: Path, out_path: Path, measured: dict):
    """
    Second pass: apply precise loudness normalization when we have measurements.
    If measurements are None, fall back to single-pass loudnorm (approximate).
    """
    if measured is None:
        # Fallback single-pass
        run_ffmpeg([
            '-y',
            '-i', str(in_path),
            '-c:v', 'copy',
            '-af', f"loudnorm=I={TARGET_I}:TP={TARGET_TP}:LRA={TARGET_LRA}:print_format=summary",
            '-c:a', 'aac', '-b:a', '192k', '-ar', '48000', '-ac', '2',
            str(out_path)
        ])
        return

    filter_str = (
        f"loudnorm=I={TARGET_I}:TP={TARGET_TP}:LRA={TARGET_LRA}"
        f":measured_I={measured['measured_I']}"
        f":measured_TP={measured['measured_TP']}"
        f":measured_LRA={measured['measured_LRA']}"
        f":measured_thresh={measured['measured_thresh']}"
        f":offset={measured['offset']}"
        f":linear=true:print_format=summary"
    )
    run_ffmpeg([
        '-y',
        '-i', str(in_path),
        '-c:v', 'copy',
        '-af', filter_str,
        '-c:a', 'aac', '-b:a', '192k', '-ar', '48000', '-ac', '2',
        str(out_path)
    ])

def normalize_to_target(in_path: Path, head_seconds: float = None) -> Path:
    """
    Create a normalized temp file for this input.
    If head_seconds is provided, first trim to that duration (from t=0), then normalize.
    If no audio, still honor trimming and then copy.
    """
    tmp_dir = Path(tempfile.mkdtemp(prefix="lnorm_"))

    # Optional pre-trim for head-only use cases (e.g., hook head)
    src_for_norm = in_path
    if head_seconds is not None:
        trim_path = tmp_dir / (in_path.stem + f"_trim_{int(head_seconds*1000)}ms.mp4")
        run_ffmpeg([
            '-y', '-i', str(in_path),
            '-t', f"{head_seconds}",
            '-c', 'copy',  # fast trim; we'll re-encode audio during loudnorm
            str(trim_path)
        ])
        src_for_norm = trim_path

    out_path = tmp_dir / (in_path.stem + "_ln.mp4")

    if not has_audio(src_for_norm):
        # Copy (and ensure we output a file even if trimmed)
        run_ffmpeg([
            '-y', '-i', str(src_for_norm),
            '-c', 'copy', str(out_path)
        ])
        return out_path

    measured = loudnorm_measure(src_for_norm)
    loudnorm_apply(src_for_norm, out_path, measured)
    return out_path

def concat_hook_ad_norm(hook_path: Path, ad_path: Path, temp_output_path: Path):
    # If TRIM_HOOK_HEAD is enabled, only keep the first HOOK_HEAD_SECONDS of the hook
    head_secs = HOOK_HEAD_SECONDS if TRIM_HOOK_HEAD else None
    hook_norm = normalize_to_target(hook_path, head_seconds=head_secs)
    ad_norm   = normalize_to_target(ad_path,   head_seconds=None)

    # IMPORTANT: Re-encode the concat to widely compatible H.264/AAC with CFR & GOP
    run_ffmpeg([
        '-i', str(hook_norm),
        '-i', str(ad_norm),
        '-filter_complex', "[0:v:0][0:a:0][1:v:0][1:a:0]concat=n=2:v=1:a=1[v][a]",
        '-map', '[v]', '-map', '[a]',
        '-r', str(FPS),             # CFR
        '-vsync', 'cfr',
        *video_encoder_args(),      # GPU or x264 (sets -g)
        *video_common_compat(),     # yuv420p + profiles + avc1
        # --- Audio: AAC LC stereo 48k ---
        '-c:a', 'aac', '-b:a', '192k', '-ar', '48000', '-ac', '2',
        '-movflags', '+faststart',
        '-y', str(temp_output_path)
    ])

    # Clean temp normalized inputs (and their trim dirs if any)
    try:
        shutil.rmtree(hook_norm.parent, ignore_errors=True)
        if ad_norm.parent != hook_norm.parent:
            shutil.rmtree(ad_norm.parent, ignore_errors=True)
    except Exception:
        pass

def append_cta_norm(in_path: Path, cta_path: Path, temp_out_path: Path):
    """
    Append a CTA clip to the end of an existing render (`in_path`).
    Both in_path and CTA are loudness-normalized sources for clean concat.
    Returns a file at `temp_out_path`.
    """
    # Normalize CTA (full clip)
    cta_norm = normalize_to_target(cta_path, head_seconds=None)

    # Concat: input (already rendered) + CTA
    run_ffmpeg([
        '-i', str(in_path),
        '-i', str(cta_norm),
        '-filter_complex', "[0:v:0][0:a:0][1:v:0][1:a:0]concat=n=2:v=1:a=1[v][a]",
        '-map', '[v]', '-map', '[a]',
        '-r', str(FPS),             # CFR
        '-vsync', 'cfr',
        *video_encoder_args(),      # GPU or x264 (sets -g)
        *video_common_compat(),     # yuv420p + profiles + avc1
        '-c:a', 'aac', '-b:a', '192k', '-ar', '48000', '-ac', '2',
        '-movflags', '+faststart',
        '-y', str(temp_out_path)
    ])

    # Clean temp
    try:
        shutil.rmtree(cta_norm.parent, ignore_errors=True)
    except Exception:
        pass

def final_loudnorm(in_path: Path, out_path: Path):
    """Apply final loudness pass on the entire render (post mix or post concat)."""
    if not has_audio(in_path):
        run_ffmpeg(['-y', '-i', str(in_path), '-c', 'copy', str(out_path)])
        return
    measured = loudnorm_measure(in_path)
    loudnorm_apply(in_path, out_path, measured)

# ===== Worker for parallel execution =====
def process_pair(hook_file: str, ad_file: str, songs_list: list, ctas_list: list,
                 export_nosong: bool, export_song_versions: bool):
    """
    Worker that handles a single (hook, ad) pair.
    If APPEND_CTA and `ctas_list` provided, renders one output per CTA variant.
    Returns (hook, ad, ok, message)
    """
    hook_path = hooks_dir / hook_file
    ad_path   = ads_dir   / ad_file

    hook_base = Path(hook_file).stem
    ad_base   = Path(ad_file).stem

    try:
        # One concat for this (hook, ad) pair
        temp_concat_path = temp_dir / f"temp_concat_{ad_base}_{hook_base}.mp4"
        output_subdir = output_dir / ad_base
        output_subdir.mkdir(parents=True, exist_ok=True)

        # HOOK + AD (your original function)
        concat_hook_ad_norm(hook_path, ad_path, temp_concat_path)

        # Decide CTA variants: if enabled and present -> use all; else -> single None variant
        cta_variants = ctas_list if (APPEND_CTA and ctas_list) else [None]

        for cta in cta_variants:
            # Where to place outputs (if multiple CTA clips, nest by CTA name)
            if cta:
                cta_base = Path(cta).stem
                variant_outdir = output_subdir / f"with-cta/{cta_base}"
            else:
                variant_outdir = output_subdir / "no-cta"
            variant_outdir.mkdir(parents=True, exist_ok=True)

            # If there is a CTA, append it to the temp_concat_path -> temp_with_cta_path
            working_input = temp_concat_path
            maybe_with_cta = temp_dir / f"temp_withcta_{ad_base}_{hook_base}{'_'+cta_base if cta else ''}.mp4"
            if cta:
                append_cta_norm(working_input, ctas_dir / cta, maybe_with_cta)
                working_input = maybe_with_cta

            # 1) Always export NO-SONG version (if configured)
            if export_nosong:
                output_subdir_no_song = variant_outdir / "no-song"
                output_subdir_no_song.mkdir(parents=True, exist_ok=True)
                suffix = f"{'_'+cta_base if cta else ''}_nosong"
                final_output_nosong = output_subdir_no_song / f"{ad_base}_{hook_base}{suffix}.mp4"
                final_loudnorm(working_input, final_output_nosong)

            # 2) Per-song background mix versions (if configured and songs available)
            if export_song_versions and songs_list:
                for song in songs_list:
                    song_path = songs_dir / song
                    song_base = Path(song).stem
                    temp_mix_path = temp_dir / f"temp_mix_{ad_base}_{hook_base}{'_'+cta_base if cta else ''}_{song_base}.mp4"
                    output_subdir_song = variant_outdir / song_base
                    output_subdir_song.mkdir(parents=True, exist_ok=True)
                    final_output_song = output_subdir_song / f"{ad_base}_{hook_base}{'_'+cta_base if cta else ''}_{song_base}.mp4"

                    if not has_audio(working_input):
                        # No audio in concat: just copy video, then normalize (which will copy)
                        shutil.copyfile(working_input, temp_mix_path)
                    else:
                        run_ffmpeg([
                            '-i', str(working_input),
                            '-stream_loop', '-1', '-i', str(song_path),
                            '-filter_complex',
                            (
                                f"[1:a]volume={SONG_BG_VOL}[bg];"
                                "[0:a][bg]amix=inputs=2:duration=first:dropout_transition=0,"
                                "alimiter=limit=0.95[aout]"
                            ),
                            '-map', '0:v:0', '-map', '[aout]',
                            '-c:v', 'copy',                        # keep the GPU-encoded video
                            '-c:a', 'aac', '-b:a', '192k', '-ar', '48000', '-ac', '2',
                            '-shortest',
                            '-movflags', '+faststart',
                            '-y', str(temp_mix_path)
                        ])

                    final_loudnorm(temp_mix_path, final_output_song)
                    try:
                        os.remove(temp_mix_path)
                    except FileNotFoundError:
                        pass

            # cleanup temp_with_cta if created
            if cta:
                try:
                    os.remove(maybe_with_cta)
                except FileNotFoundError:
                    pass

        # Cleanup base concat temp
        try:
            os.remove(temp_concat_path)
        except FileNotFoundError:
            pass

        return (hook_file, ad_file, True, "ok")

    except subprocess.CalledProcessError as e:
        return (hook_file, ad_file, False, f"ffmpeg error: {e}")
    except Exception as e:
        return (hook_file, ad_file, False, f"unexpected error: {e}")

def preprocess_inputs():
    """Ensure inputs are 1080x1920 (sequential to avoid extra disk contention)."""
    for folder in (hooks_dir, ads_dir, ctas_dir):
        if not folder.exists():
            continue
        for p in folder.iterdir():
            if p.is_file() and p.suffix.lower() == '.mp4':
                ensure_1080x1920(p)

def main():
    # ===== Preprocess to 1080x1920 =====
    preprocess_inputs()

    # ===== Gather media =====
    hooks = list_media(hooks_dir, {'.mp4'})
    ads   = list_media(ads_dir,   {'.mp4'})
    songs = list_media(songs_dir, {'.mp3', '.m4a', '.wav'})
    ctas  = list_media(ctas_dir,  {'.mp4'}) if APPEND_CTA and ctas_dir.exists() else []

    songs_present = len(songs) > 0

    if not hooks or not ads:
        print("[info] Nothing to do (no hooks or no ads).")
        return

    pairs = list(itertools.product(hooks, ads))
    total = len(pairs)
    max_workers = 8

    print(f"[start] {total} pairs | songs_present={songs_present} "
          f"| EXPORT_NOSONG={EXPORT_NOSONG_VERSION} | EXPORT_SONGS={EXPORT_SONG_VERSIONS} "
          f"| APPEND_CTA={APPEND_CTA} | CTA_VARIANTS={len(ctas)} | workers={max_workers}")

    # Build tasks
    futures = []
    done_ok = 0
    done_err = 0

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for hook, ad in pairs:
            futures.append(
                ex.submit(
                    process_pair,
                    hook, ad,
                    songs if songs_present else [],
                    ctas,
                    EXPORT_NOSONG_VERSION,
                    (EXPORT_SONG_VERSIONS and songs_present)
                )
            )

        for i, fut in enumerate(as_completed(futures), start=1):
            hook_file, ad_file, ok, msg = fut.result()
            prefix = "[ok] " if ok else "[err]"
            if ok:
                done_ok += 1
            else:
                done_err += 1
            print(f"{prefix} {i}/{total}  ad={ad_file}  hook={hook_file}  -> {msg}")

    print(f"[done] Completed. ok={done_ok}, errors={done_err} "
          f"({'hook head only' if TRIM_HOOK_HEAD else 'full hook'}; "
          f"Encoder={VIDEO_ENCODER}; H.264 High@L4.0 yuv420p, CFR {FPS}fps).")

if __name__ == "__main__":
    main()
