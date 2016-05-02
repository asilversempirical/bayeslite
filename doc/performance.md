# BQL performance

This is an analysis of the questions raised in the performance expectations
section of ["Bayeslite Usability"][usability].

## SIMULATE

### Expected performance

Under the basic crosscat model, initialization time per model for a conditional
simulation query is linear in the number of categories attached to each view
containing a condition. Once the posterior distribution on the categories has
been calculated from the simulation conditions, time to sample a row is a linear
combination of... anyway, we expect this to be fast, and it is fast.

As of [current bayeslite] and [current crosscat], a `SIMULATE` command is run in
either [`compile_simulate`] or [`execute_phrase`]. This code is heavily
duplicated... we should [fix that]... In each case, a table is created during
the compile phase, the difference being that in the `compile_simulate` case the
table is temporary. They work by a sleight of hand: the simulations are drawn
_during the compile phase_, by a call to [`bayesdb_simulate`]and dropped into
the table. This calls the metamodel's `simulate_joint` method, which in
[crosscat's case] calls the [`Engine.simple_predictive_sample`] method, which
calls [`_do_simple_predictive_sample`], and that calls
[`simple_predictive_sample_multistate`] (there is a single-state option which
skips over this call, but we never use that at the moment).

`simple_predictive_sample_multistate` is problematic, in that it draws samples
from each model uniformly. For conditional draws,
[this is inaccurate.]

Anyway, it calls `sample_utils.simple_predictive_sample`, which calls
`simple_predictive_observed` or `simple_predictive_unobserved`...


[usability]: https://docs.google.com/document/d/1LX3krkRKz5WeykHrYftxOGGYl_GaRTD_QGwDgham-Pc/edit#
[current bayeslite]: https://github.com/probcomp/bayeslite/tree/9e09a45da56aff8fd0043c6e63f8c04f4900d582
[current crosscat]: https://github.com/probcomp/crosscat/tree/4f75431b06978c77fc1c8e9c559af0f68101316d
[`compile_simulate`]: https://github.com/probcomp/bayeslite/blob/55e8c3b59f3da480308f0b26ef39eb901dac1d09/src/compiler.py#L563
[`execute_phrase`]: https://github.com/probcomp/bayeslite/blob/55e8c3b59f3da480308f0b26ef39eb901dac1d09/src/bql.py#L89
[fix that]: https://github.com/empiricalsys/bayeslite/issues/9
[`bayesdb_simulate`']: https://github.com/probcomp/bayeslite/blob/55e8c3b59f3da480308f0b26ef39eb901dac1d09/src/bqlfn.py#L362
[crosscat's case]: https://github.com/probcomp/bayeslite/blob/55e8c3b59f3da480308f0b26ef39eb901dac1d09/src/metamodels/crosscat.py#L1117
[`simple_predictive_sample`]: https://github.com/probcomp/crosscat/blob/4f75431b06978c77fc1c8e9c559af0f68101316d/src/LocalEngine.py#L297
[`_do_simple_predictive_sample`]: https://github.com/probcomp/crosscat/blob/4f75431b06978c77fc1c8e9c559af0f68101316d/src/LocalEngine.py#L882
[`simple_predictive_sample_multistate`]: https://github.com/probcomp/crosscat/blob/4f75431b06978c77fc1c8e9c559af0f68101316d/src/utils/sample_utils.py#L310
[`simple_predictive_sample`]: https://github.com/probcomp/crosscat/blob/4f75431b06978c77fc1c8e9c559af0f68101316d/src/utils/sample_utils.py#L278
[this is inaccurate.]: https://github.com/probcomp/crosscat/issues/97
