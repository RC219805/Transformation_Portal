The canonical branded loop remains at `public/video/dna-loop.mp4`.

The legacy `public/video/login-loop.mp4` file can remain as a local placeholder,
but homepage and login reference `dna-loop.mp4` as the stable public asset
path. Both pages use static neutral surfaces. Their dormant video elements
have `preload="none"`, omit autoplay, and remain hidden, so public entry does
not download decorative video or introduce background motion.
