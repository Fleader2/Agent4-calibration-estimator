"""Loading/compiling an Antimony model into a runnable simulation engine.

Identical engine choice and rationale to ``app.agent3.loader`` on the sibling
``agent3-simulation-diagnostics`` repository -- re-implemented independently here (never
imported) per this project's own decoupling convention, and per the task's own explicit
instruction to "consume or reproduce Agent 3's RoadRunner simulation semantics rather than
introducing a second simulation engine." Uses ``antimony`` (Antimony -> SBML translation) and
``libroadrunner`` (SBML -> ODE simulation, imported as ``roadrunner``) directly -- not the
heavier ``tellurium`` convenience meta-package.

A compilation failure is never raised as a Python exception here; it is reported as data
(``ModelLoadResult``), exactly as Agent 3's own loader does.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import antimony
import roadrunner

#: Mirrors ``app.agent3.loader``'s own identically-reasoned lock: ``antimony``'s module-level
#: parser state is a single, process-global, non-reentrant resource.
_ANTIMONY_LOCK = threading.Lock()


@dataclass(frozen=True, slots=True)
class ModelLoadResult:
    succeeded: bool
    roadrunner_instance: roadrunner.RoadRunner | None
    sbml_text: str | None
    message: str | None
    roadrunner_version: str


def load_antimony_model(antimony_text: str) -> ModelLoadResult:
    """Compile ``antimony_text`` into a loaded ``roadrunner.RoadRunner`` instance, or report
    exactly why compilation failed. Never mutates ``antimony_text``, never retries with a
    "repaired" variant."""
    with _ANTIMONY_LOCK:
        antimony.clearPreviousLoads()
        antimony.freeAll()
        load_code = antimony.loadAntimonyString(antimony_text)
        if load_code < 0:
            message = antimony.getLastError()
            antimony.clearPreviousLoads()
            return ModelLoadResult(
                succeeded=False,
                roadrunner_instance=None,
                sbml_text=None,
                message=f"Antimony compilation failed: {message}",
                roadrunner_version=roadrunner.__version__,
            )

        main_module = antimony.getMainModuleName()
        try:
            sbml_text = antimony.getSBMLString(main_module)
        except Exception as exc:  # pragma: no cover - defensive; not expected for valid antimony
            antimony.clearPreviousLoads()
            return ModelLoadResult(
                succeeded=False,
                roadrunner_instance=None,
                sbml_text=None,
                message=f"Antimony->SBML conversion failed for module {main_module!r}: {exc}",
                roadrunner_version=roadrunner.__version__,
            )
        antimony.clearPreviousLoads()

    if not sbml_text:
        return ModelLoadResult(
            succeeded=False,
            roadrunner_instance=None,
            sbml_text=None,
            message="Antimony compiled with no error, but produced an empty SBML document.",
            roadrunner_version=roadrunner.__version__,
        )

    try:
        rr = roadrunner.RoadRunner(sbml_text)
    except Exception as exc:
        return ModelLoadResult(
            succeeded=False,
            roadrunner_instance=None,
            sbml_text=sbml_text,
            message=f"RoadRunner rejected the compiled SBML: {exc}",
            roadrunner_version=roadrunner.__version__,
        )

    return ModelLoadResult(
        succeeded=True,
        roadrunner_instance=rr,
        sbml_text=sbml_text,
        message=None,
        roadrunner_version=roadrunner.__version__,
    )


__all__ = ["ModelLoadResult", "load_antimony_model"]
