| Component | What it does | How many |
| --- | --- | --- |
| Website | Takes the video and shows the plays as they are cut | 1 |
| Worker | Sends each request to the website, or to the cutter for that film | 1 |
| Cutter | Keeps one film in order. Starts the next job. Writes play 1, then play 2, then play 3 | 1 per film |
| Ingest | Saves the full video to R2 | 1 per film |
| Labeler | Decides what one 40-second stretch is: scrimmage, special teams, junk, or uncertain | Many per film |
| Cut | Writes one settled play to R2 as its own clip | 1 per play |
| Person | Marks any stretch that is still uncertain after a second look | As needed |
| Analyzer | Charts one kept play from its clip. Starts after the cut is final | 1 per kept play, later |
| D1 | Stores the project, the spans, and the plays | 1, already created |
| R2 | Stores the full video and every play clip | 1 bucket, already created |

The labelers and the cutter are the cut. Labelers read the long video in parallel. The cutter waits until the start of the film has settled, then sends one cut job per play, so the clips come out in order. Uncertain stretches get one more labeling pass. Whatever is still uncertain waits for a person. Scrimmage and special teams are kept. Junk is dropped.

The analyzers are the second stage. Each one reads a single play clip and writes that play's chart. They start after the kept plays are final.

On Cloudflare the website, cutter, and tasks are containers from this image. The worker is the public URL. Ingest, label, and cut are the `film-task` command. The cutter is `film-cutter`. The website is `film-analysis`.
