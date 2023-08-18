# Using TkAgg backend because using of cv2 and matplotlib in the same script
import matplotlib

matplotlib.use("TkAgg")

from src.Backend.Funcs import loadGeneralYaml, load_yaml

import matplotlib.pyplot as plt
import numpy as np
import pathlib
import sys

PATH = pathlib.Path(__file__).parent.absolute()
CONTROLS = [
    ["1/dij", "dij"],
    [
        "-P_gij * gij*",
        "gij-gij*",
        "(-P_gij * gij*) + (gij - gij*)",
        "-P_gij * gij* with rotation",
        "gij - gij* with rotation",
        "(-P_gij * gij*) + (gij - gij*) with rotation",
    ],
]
COLORS = ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray", "olive"]
GEN_YAML = loadGeneralYaml(PATH)
DRONE_YAML = load_yaml(PATH)
TITLE = f"Summary {'leader' if GEN_YAML['Leader_Follower'] == 0 else 'follower'} drone with control ({CONTROLS[GEN_YAML['Leader_Follower']][DRONE_YAML['control']-1]})"
DRONE = sys.argv[1] if len(sys.argv) > 1 else 1

plt.rcParams["figure.autolayout"] = True

if (integ := np.loadtxt(f"{PATH}/out/drone_{DRONE}_int.txt")).size in (0, 1):
    integ = None
    fig, ax = plt.subplots(3, 1, figsize=(10, 7), sharex=True, num=TITLE)
else:
    fig, ax = plt.subplots(2, 2, figsize=(10, 7), sharex=True, num=TITLE)

fig.suptitle(TITLE, fontsize=14)
ax = ax.reshape(-1)

err = np.loadtxt(f"{PATH}/out/drone_{DRONE}_error.txt")
err_pix = np.loadtxt(f"{PATH}/out/drone_{DRONE}_errorPix.txt")
time = np.loadtxt(f"{PATH}/out/drone_{DRONE}_time.txt")
vels = np.loadtxt(f"{PATH}/out/drone_{DRONE}_input.txt")

vx = vels[:, 0]
vy = vels[:, 1]
vz = vels[:, 2]
vroll = vels[:, 3]
vpitch = vels[:, 4]
vyaw = vels[:, 5]

if err.size == 1:
    print("There's no data for this drone")
    sys.exit()

print(
    f"""
Drone {GEN_YAML['Leader_Follower']} with control ({CONTROLS[GEN_YAML['Leader_Follower']][DRONE_YAML['control']-1]})
Tiempo total -> {time[-1]:5f}
{f'Error final -> {err[-1]:5f} -- Max error -> {np.max(err[:], axis=0):5f}' if len(err.shape) == 1 else f'Error final -> ({err[-1, 0]:5f}, {err[-1, 1]:5f}, {err[-1, 2]:5f}) -- Max error -> ({np.max(err[:, 0], axis=0):5f}, {np.max(err[:, 1], axis=0):5f}, {np.max(err[:, 2], axis=0):5f})'}
Velocidad final -> ({vx[-1]:5f}, {vy[-1]:5f}, {vz[-1]:5f}, {vyaw[-1]:5f})
Promedio de tiempo por frame -> {np.mean(time[1:] - time[:-1]):5f}
"""
)

#########################  FIRST PLOT ERROR  #########################

ax[0].title.set_text(f"Error")
if len(err.shape) == 1:
    ax[0].step(
        time[:],
        err[:],
        "-",
        color="blue",
        label=f"Error (c)",
        where="post",
    )
    ax[0].plot(
        [time[1], time[-1]],
        [err[-1], err[-1]],
        "--",
        color="blue",
        label=f"y={err[-1]:.3f}",
        alpha=0.5,
    )
else:
    for i in range(3):
        ax[0].step(
            time[:],
            err[:, i],
            "-",
            color=COLORS[i],
            label=f"Error {'x' if i == 0 else ('y' if i == 1 else 'z')} (c)",
            where="post",
        )
        ax[0].plot(
            [time[1], time[-1]],
            [err[-1, i], err[-1, i]],
            "--",
            color=COLORS[i],
            label=f"y={err[-1, i]:.3f}",
            alpha=0.5,
        )
    ax[0].step(
        time[:],
        np.linalg.norm(err[:, :], axis=1),
        "-",
        color="k",
        label=f"Total Error (c)",
        where="post",
    )
    ax[0].plot(
        [time[1], time[-1]],
        [temporalErrorNorm := np.linalg.norm(err[-1, :]), temporalErrorNorm],
        "--",
        color="k",
        label=f"y={temporalErrorNorm:.3f}",
        alpha=0.5,
    )
if np.any(err_pix != 0):
    ax[0].plot(time[:], err_pix[:], "-", c="r", label="Error (px)")
    ax[0].plot(
        [time[1], time[-1]],
        [err_pix[-1], err_pix[-1]],
        "--",
        c="r",
        label=f"y={err_pix[-1]:.3f}",
        alpha=0.25,
    )
ax[0].plot(
    [time[1], time[-1]],
    [0, 0],
    "--",
    c="k",
    label=f"y=0",
    alpha=0.25,
)

box = ax[0].get_position()
ax[0].set_position([box.x0, box.y0, box.width * 0.99, box.height])
ax[0].legend(loc="center left", bbox_to_anchor=(1, 0.5), shadow=True)
ax[0].set_ylabel("Error Promedio")


#########################  SECOND PLOT VELOCITIES  #########################
ax[1].title.set_text("Velocities")
ax[1].plot(
    [time[1], time[-1]],
    [0, 0],
    "--",
    c="k",
    label=f"y=0",
    alpha=0.25,
)
ax[1].step(time[:], vx[:], "-", label="$V_x$ (m/s)")
ax[1].step(time[:], vy[:], "-", label="$V_y$ (m/s)")
ax[1].step(time[:], vz[:], "-", label="$V_z$ (m/s)")
ax[1].step(time[:], vroll[:], "-", label="$W_x$ (rad/s)", alpha=0.5)
ax[1].step(time[:], vpitch[:], "-", label="$W_y$ (rad/s)", alpha=0.5)
ax[1].step(time[:], vyaw[:], "-", label="$W_z$ (rad/s)")
ax[1].legend(loc="center left", bbox_to_anchor=(1, 0.5))
ax[1].set_ylabel("Velocidades")


#########################  THIRD PLOT GAINS  #########################
Gains = [[[], []], [[], []]]
ax[2].title.set_text("Adaptative gains")
for index_j, j in enumerate(["v", "w"]):
    for index_i, i in enumerate(["kp", "ki"]):
        try:
            lamb = np.loadtxt(f"{PATH}/out/drone_{DRONE}_{j}_{i}.txt")
            Gains[index_j][index_i] = lamb
            if len(lamb.shape) == 1:
                ax[2].plot(
                    time[:],
                    lamb[:],
                    label=j + ": $\lambda_{" + i + "}$",
                    color=COLORS[index_j * 3 + index_i],
                )
                ax[2].plot(
                    [time[1], time[-1]],
                    [lamb[-1], lamb[-1]],
                    "--",
                    color=COLORS[index_j * 3 + index_i],
                    label=f"y={lamb[-1]:.3f}",
                    alpha=0.25,
                )
            else:
                ax[2].plot(
                    time[:],
                    np.mean(lamb[:], axis=1),
                    label=j + ": $\lambda_{" + i + "}$",
                    color=COLORS[index_j * 3 + index_i],
                )
                ax[2].plot(
                    [time[1], time[-1]],
                    [tempMean := np.mean(lamb[-1]), tempMean],
                    "--",
                    color=COLORS[index_j * 3 + index_i],
                    label=f"y={tempMean:.3f}",
                    alpha=0.25,
                )
        except Exception as e:
            pass

ax[2].legend(loc="center left", bbox_to_anchor=(1, 0.5))
ax[2].set_ylabel("Lambda")
ax[2].set_xlabel("Tiempo (s)")


#########################  FOURTH PLOT INTEGRALS  #########################
if integ is not None:
    ax[3].title.set_text("Applied Integrals")
    ax[3].plot(
        time[:],
        integ[:, 0] * Gains[0][1][:, 0],
        label="$I_x * \lambda_{ki}$",
        alpha=0.7,
    )
    ax[3].plot(
        time[:],
        integ[:, 1] * Gains[0][1][:, 1],
        label="$I_y * \lambda_{ki}$",
        alpha=0.7,
    )
    ax[3].plot(
        time[:],
        integ[:, 2] * Gains[0][1][:, 2],
        label="$I_z * \lambda_{ki}$",
        alpha=0.7,
    )
    ax[3].plot(
        [time[1], time[-1]],
        [0, 0],
        "--",
        c="k",
        label=f"y=0",
        alpha=0.25,
    )
    ax[3].legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax[3].set_ylabel("Integrales")

#########################  TIGHT LAYOUT, SAVING AND SHOWING  #########################
# plt.get_current_fig_manager().full_screen_toggle()
# fig.tight_layout()
# fig.savefig(f"{PATH}/out/out_velocities.png", bbox_inches="tight", pad_inches=0.1)

# fig2, ax2 = plt.subplots(2, 2, figsize=(10, 7), num=f"Angles", sharex=True)
# time = np.loadtxt(f"{PATH}/out/drone_{DRONE}_toAnglesTime.txt")
# angles1 = np.rad2deg(np.loadtxt(f"{PATH}/out/drone_{DRONE}_toAngles1.txt"))
# angles2 = np.rad2deg(np.loadtxt(f"{PATH}/out/drone_{DRONE}_toAngles2.txt"))
# time = time[:].reshape(-1, 2)


# maxangle = np.min([angles1.shape[0], angles2.shape[0], time.shape[0]])
# maxtime = np.min([angles1.shape[0], angles2.shape[0], time.shape[0]])


# ax2[0, 0].title.set_text("Angles 1")
# ax2[0, 0].plot(time[:maxangle, 0], angles1[:maxtime, 0], "-", label="pitch - R1", color="blue")
# ax2[0, 0].plot(time[:maxangle, 0], angles1[:maxtime, 1], "-", label="yaw - R1", color="orange")
# ax2[0, 0].plot(time[:maxangle, 0], angles1[:maxtime, 2], "-", label="roll - R1", color="green")
# ax2[1, 0].title.set_text("Angles 2")
# ax2[1, 0].plot(time[:maxangle, 0], angles1[:maxtime, 3], "-", label="yaw - sent", color="orange")
# ax2[0, 0].legend(loc="best", bbox_to_anchor=(1, 0.5))
# ax2[1, 0].legend(loc="best", bbox_to_anchor=(1, 0.5))

# ax2[0, 1].title.set_text("Angles 1")
# ax2[0, 1].plot(time[:maxangle, 0], angles2[:maxtime, 0], "-", label="pitch - R1", color="blue")
# ax2[0, 1].plot(time[:maxangle, 0], angles2[:maxtime, 1], "-", label="yaw - R1", color="orange")
# ax2[0, 1].plot(time[:maxangle, 0], angles2[:maxtime, 2], "-", label="roll - R1", color="green")
# ax2[1, 1].title.set_text("Angles 2")
# ax2[1, 1].plot(time[:maxangle, 0], angles2[:maxtime, 3], "-", label="yaw - sent", color="orange")
# ax2[0, 1].legend(loc="best", bbox_to_anchor=(1, 0.5))
# ax2[1, 1].legend(loc="best", bbox_to_anchor=(1, 0.5))

plt.show()
