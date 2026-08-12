# Verified offline demo

Command:

```bash
python cli.py --topic "How AI helps creators" --duration 16 --language en
```

Verification result:

- 15/15 unit, automation, and FFmpeg integration tests passed.
- The workflow completed with status `completed`.
- All scenes had valid assets.
- Expected duration: 16.307 seconds.
- Actual duration: 16.303 seconds.
- Duration delta: 0.004 seconds.

`demo_final.mp4` is the original offline baseline sample. The current workflow additionally burns subtitles into the final MP4 and accepts mixed image/video assets. A video-material example can be generated with:

```bash
python cli.py --topic "AI creator workflow" --duration 12 --language en --assets-dir examples/mixed_assets
```

The visual selection is deliberately lightweight and local; semantic visual scoring, timeline editing, and generation models remain future enhancements.

Stage-two shot detection and vertical crop demo:

```bash
python cli.py --topic "AI creator workflow" --duration 12 --language en \
  --assets-dir examples/stage2_assets --aspect-ratio 9:16 --fit-mode crop \
  --scene-threshold 0.15 --transition-seconds 0.35
```
