"""Readable pages made from stored results; no model, GPU, or network.

Each page is one self-contained HTML file and a pure function of the stored
JSON it is given: the same bytes in give the same bytes out. Nothing here
computes a new statistic. A page shows what the engine stored, and says so
where a value was not stored.

Layout, so a later page (a study's results, say) reuses the general parts:

* ``page``    the document frame, styles, tables, and small building blocks;
* ``charts``  inline SVG line charts and forest charts, with a text alternative;
* ``jlens_assessment``  the J-lens assessment page, built from those two;
* ``science_report``    the ``science report`` verb: find the stored report,
  choose where the page goes, and write it without touching a run directory;
* ``study_results``     the results page of an ordinary study, drawn from the
  same reading ``results export`` makes (``results_export.read_results``).
"""
