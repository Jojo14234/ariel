from a3_exp.lib import Experiment, SPAWN_RUGGED
import numpy as np

class SimpleViewer(Experiment):

    def main(self):
        self.init_nde_hpd()
        rng = np.random.default_rng(42)

        genome = rng.normal(size=64 * 3)
        graph = self._genotype_to_graph(list(genome.reshape(3, 64).astype(np.float32)))
        spec = self._graph_to_mj_spec(graph)
        model = self.spec_to_olympic_world(spec)
        self.view(model, lambda _, d: d.ctrl, self.fitness)


if __name__ == '__main__':
    SimpleViewer().main()
