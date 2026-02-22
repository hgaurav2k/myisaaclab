import mujoco
import mujoco.viewer


def visualize_station():
    """Load and visualize the station.xml MuJoCo model"""

    # Get the path to the station.xml file
    station_xml_path = "station_mjcf/station.xml"

    model = mujoco.MjModel.from_xml_path(str(station_xml_path))
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    visualize_station()
