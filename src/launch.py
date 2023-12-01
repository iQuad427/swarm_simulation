from src.experiments import (
    SameSpeedRandomPlacementDelaunayTriangulationGlobalCommunicationExperiment,
    AllStaticButOneRandomPlacementDelaunayTriangulationDistanceLimitedCommunicationExperiment,
    AllStaticRectanglePlacementExperiment, DelaunayCommunicationExperiment, TestExperiment, OneAwayFromOther,
)
from src.modules.triangulation.types.reconstruct import ReconstructTriangulation

if __name__ == '__main__':
    # TODO:
    #  Simulation Functionalities:
    #   - show the communication radius of each agent;
    #   - give the arena shape and size as a parameter of the simulation;
    #   - add a way to save the triangulation over time (dashboard?);
    #      - save triangulation and agents positions at each x seconds;
    #      - then be able to plot the evolution of the triangulation over time;
    #   - add parametrisation inside of the GUI;
    #      - may require to separate the GUI from the simulation;
    #         - first, be able to run the simulation without the GUI;
    #         - then, allow for start/stop of the simulation, with thread kill and start;
    #         - finally, add the GUI and add the parametrisation;
    #      - idea is to allow for changing the simulation, without having to stop the UI;
    #      - parameters to change:
    #         - number of agents;
    #         - speed of the agents;
    #         - precision of the triangulation;
    #         - frequency of the communication;
    #         - shape and size of the arena;
    #      - some require to stop the simulation, some don't;
    #  Specific Functionalities:
    #   - modify the communication to be more realistic;
    #      - more chances to communicate with close agents
    #      - add noise to the distances (about 10 centimeters of precision)
    #      - add chance for bad package reception (negative distance, distance of 0, etc.)
    #   - possibly add the TTL mechanism to avoid overwriting new information with old ones
    #      - already implemented TTL before sending information
    #      - still need to add a TTL verification at use also (could even remove the TTL before sending)
    #         - need to transfer the relative age of the data
    #   - find an iterative way to triangulate the swarm;
    #      - avoid recomputing the triangulation from scratch each time

    experiment = OneAwayFromOther(
        # dim=5,  # number of agents
        refresh_rate=0.01,  # refresh rate of the simulation (in seconds)
        agents_speed=1,  # speed of the agents (in meters per second)
        triangulation_precision=0.1,  # precision of the triangulation (in meters)
        communication_frequency=0.01,  # frequency of the communication (in seconds)
        communication_radius=15,  # radius of the communication (in meters)
    )
    experiment.launch()
