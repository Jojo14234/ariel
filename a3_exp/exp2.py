import mujoco as mj
from a3_exp.main import MainExperiment, ExpConfig
from a3_exp.strategies import CMAES


class Exp2(MainExperiment):

    def testing_cma(self):
        spec = self.gecko_spec()
        es = CMAES(3, 100, 42)

        for ig in range(30):
            genomes = es.ask()
            models = [self.spec_to_olympic_world(spec, list(g)) for g in genomes]
            self.run_inner(
                mj_model,
            strategy_cls: str,
            strat_kw: Dict,
            policy_cls: str,
            population_size: int,
            n_generations: int,
            sim_duration: int,
            sim_steps_per_cycle: int,
            fitness: Callable,

            )

    @staticmethod
    def fitness(_: mj.MjModel, mj_data: mj.MjData):
        return .8 + mj_data.geom('robot-core').xpos[0]


if __name__ == '__main__':
    _main_config = ExpConfig(
        nde_seed=42,
        world=1,
        outer_strategy_cls="CMA",
        outer_strat_kw=dict(seed=42),
        # outer_strat_kw=dict(seed=16, mutation_rate=.1, crossover_rate=.7, tournament_size=3),
        outer_population=10,
        outer_generations=400,
        inner_strategy_cls="CMA",
        inner_strat_kw=dict(seed=16),
        inner_population=16,
        inner_generations=10,
        inner_policy_cls="NNPolicy",
        sim_duration=10,
        sim_steps_per_cycle=10,
    )
    Exp2().run("_exp2", _main_config)
