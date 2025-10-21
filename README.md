# Remote-Microgrid-Sizing-Optimization-PSO
The algorithm performs microgrid sizing optimization by using a nested particle swarm optimization (PSO) algorithm. The microgrids considered consist of an electrical load, a solar panel array, a battery energy storage system (BESS) and a diesel generator. The provided algorithm performs the sizing of the solar panel array and the BESS while the diesel generator is assumed to be equal to the maximum instantaneous load. Future versions might include the sizing of the diesel generator as well.
The algorithm allows to choose between two battery degradation models - an energy-throughput model and a semi-empirical degradation model by Schimpe et al. [1] The algorithm also allows to choose to consider or not to consider inverter degradation, which is based on the model described in [2] by Alpízar-Castillo.
It is possible to run the sizing optimization with a rule-based dispatch for explorative purposes. The rule-based dispatch is based on the paper by Cicilo et al. [3]




[1] https://iopscience.iop.org/article/10.1149/2.1181714jes/meta
[2] https://ieeexplore.ieee.org/abstract/document/10726347
[3] https://ieeexplore.ieee.org/document/8895953?signout=success
