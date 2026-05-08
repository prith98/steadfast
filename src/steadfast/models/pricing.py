"""Cost computation table — per ``(provider, model)``, USD per million input/output tokens.

Used by ``BaseModelClient`` to attach a cost attribute to every span and to
the run manifest. Pricing snapshots are dated; reproductions should record
the snapshot date in the manifest.

Implementation in ``docs/WEEK_1.md`` §"Tuesday".
"""

from __future__ import annotations
