# Fantasy Super Rugby Pacific 2026 — solver configuration.
#
# Squad: exactly 15 players, all of whom score (no bench).
# Budget: 100m (stored raw in the data as 100_000_000).
# Max 4 players from any one Super Rugby Pacific club.
# Captain scores 2x points.
#
# `loose_forward` is the data label for #8 + flanker combined.
# `outside_back` is the data label for full-back + winger combined.
# These two are already merged upstream, so the solver treats them as ordinary
# positions with their own exact counts.

# Raw cost units in the bronze/predictions data: 1m = 1_000_000.
STARTING_BUDGET = 100_000_000

SQUAD_SIZE = 15

MAX_PLAYERS_PER_CLUB = 4

CAPTAIN_MULTIPLIER = 2
CAPTAIN_BONUS = 0

# Exact-count requirements per position. Sum == SQUAD_SIZE.
POSITION_COUNTS = {
    "prop":          2,
    "hooker":        1,
    "lock":          2,
    "loose_forward": 3,
    "scrum_half":    1,
    "fly_half":      1,
    "center":        2,
    "outside_back":  3,
}

POSITIONS = set(POSITION_COUNTS.keys())

# Boosters (Triple Captain, Limitless, Co-Captains) are once-per-season,
# single-round decisions. They are not modelled in the per-round optimiser;
# callers should override solver kwargs (e.g. budget=inf for Limitless,
# captain_multiplier=3 for Triple Captain) for the round they're played.
