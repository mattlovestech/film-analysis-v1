"""Film-reading agents.

`PlayBoundaryAgent` watches an uploaded video or a public YouTube link and returns play timestamps.
`FilmAgent` reads plays that are already charted.
"""

from film_analysis.agents.play_boundaries import PlayBoundaryAgent
from film_analysis.agents.protocol import AgentFinding, FilmAgent, FilmContext
from film_analysis.agents.segments import PlaySegment

__all__ = [
    "AgentFinding",
    "FilmAgent",
    "FilmContext",
    "PlayBoundaryAgent",
    "PlaySegment",
]
