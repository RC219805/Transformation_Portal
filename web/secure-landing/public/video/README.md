The front door now serves its canonical branded loop from
`public/video/dna-loop.mp4`.

The legacy `public/video/login-loop.mp4` file can remain as a local placeholder,
but homepage and login now reference `dna-loop.mp4` as the stable public asset
path. The page still falls back to the gradient treatment if the video is
removed or the browser blocks playback.
