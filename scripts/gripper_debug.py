import matplotlib

matplotlib.use("TkAgg")
import numpy as np
import matplotlib.pyplot as plt

from assets.grippers.robotiq import RobotiqGripper


def main():
    gripper = RobotiqGripper(root_folder="")
    plot_gripper(gripper)
    return gripper


def plot_gripper(gripper):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for mesh in gripper.get_finger_meshes():
        ax.plot_trisurf(
            mesh.vertices[:, 0],
            mesh.vertices[:, 1],
            mesh.vertices[:, 2],
            triangles=mesh.faces,
            color="steelblue",
            alpha=0.5,
            edgecolor="k",
            linewidth=0.1,
        )

    origins, directions = gripper.get_closing_rays(np.eye(4))
    arrow_length = 0.02
    ax.quiver(
        origins[:, 0],
        origins[:, 1],
        origins[:, 2],
        directions[:, 0],
        directions[:, 1],
        directions[:, 2],
        length=arrow_length,
        normalize=True,
        color="red",
        linewidth=1,
    )

    ax.set_title("Gripper Closing Rays and Finger OBBs")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.auto_scale_xyz([-0.1, 0.1], [-0.1, 0.1], [0, 0.12])
    plt.tight_layout()

    z_min, z_max = gripper.standoff_range
    x_plane = np.linspace(-0.05, 0.05, 10)
    y_plane = np.linspace(-0.05, 0.05, 10)
    X, Y = np.meshgrid(x_plane, y_plane)

    base = gripper.get_base_mesh()
    ax.plot_trisurf(
        base.vertices[:, 0],
        base.vertices[:, 1],
        base.vertices[:, 2],
        triangles=base.faces,
        color="gray",
        alpha=0.3,
        linewidth=0.1,
        edgecolor="k",
    )

    for z in [z_min, z_max]:
        Z = np.full_like(X, z)
        ax.plot_surface(X, Y, Z, color="green", alpha=0.2, edgecolor="none")

    ax.text(0, 0, z_min, "Standoff min", color="green")
    ax.text(0, 0, z_max, "Standoff max", color="green")

    # Add red sphere at center origin (0, 0, 0)
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    radius = 0.01  # Adjust radius as needed
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color="red", alpha=1.0)

    # Add blue sphere at origin + gripper.tcp_offset with half the size
    if hasattr(gripper, "tcp_offset"):
        offset_pos = gripper.tcp_offset
        small_radius = radius / 2
        x_offset = small_radius * np.outer(np.cos(u), np.sin(v)) + offset_pos[0]
        y_offset = small_radius * np.outer(np.sin(u), np.sin(v)) + offset_pos[1]
        z_offset = (
            small_radius * np.outer(np.ones(np.size(u)), np.cos(v)) + offset_pos[2]
        )
        ax.plot_surface(x_offset, y_offset, z_offset, color="blue", alpha=1.0)
        ax.text(
            offset_pos[0],
            offset_pos[1],
            offset_pos[2],
            "TCP Offset",
            color="blue",
            fontsize=8,
        )

    plt.show()


if __name__ == "__main__":
    gripper = main()
