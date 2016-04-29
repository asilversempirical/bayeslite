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

As of [current bayeslite] and [current crosscat], a `SIMULATE` command


[usability]: https://docs.google.com/document/d/1LX3krkRKz5WeykHrYftxOGGYl_GaRTD_QGwDgham-Pc/edit#
[current bayeslite]: https://github.com/alxempirical/bayeslite/tree/9e09a45da56aff8fd0043c6e63f8c04f4900d582
[current crosscat]: https://github.com/probcomp/crosscat/tree/4f75431b06978c77fc1c8e9c559af0f68101316d
