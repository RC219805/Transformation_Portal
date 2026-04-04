The standalone portal serves its canonical branded loop from
`public/video/dna-portal-video-2.mp4`.

The portal UI requests this asset through the cache-friendly same-origin route
`/portal/video/dna-portal-video-2.mp4`, so both the direct backend surface and
the managed front door can reuse the same stable asset path.
