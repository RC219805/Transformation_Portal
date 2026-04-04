The standalone portal serves its canonical branded loop from
`public/video/dna-portal-video-2.mp4`.

The portal UI requests this asset through the authenticated same-origin route
`/v1/portal/video/dna-portal-video-2.mp4`, so the backend can keep the video
path stable while preserving the portal's existing access controls.
