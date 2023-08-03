# Using TkAgg backend because using of cv2 and matplotlib in the same script
import matplotlib

matplotlib.use("TkAgg")

from src.Aux.Funcs import loadGeneralYaml, load_yaml

import matplotlib.pyplot as plt
import numpy as np
import pathlib
import sys


PATH = pathlib.Path(__file__).parent.absolute()
CONTROLS = [["1/dij", "dij"], ["-P_gij * gij*", "gij-gij*"]]
COLORS = ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray", "olive"]
GEN_YAML = loadGeneralYaml(PATH)
DRONE_YAML = load_yaml(PATH)
TITLE = f"Summary {'leader' if GEN_YAML['Leader_Follower'] == 0 else 'follower'} drone with control ({CONTROLS[GEN_YAML['Leader_Follower']][DRONE_YAML['control']-1]})"
DRONE = sys.argv[1] if len(sys.argv) > 1 else 1


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
vx = np.loadtxt(f"{PATH}/out/drone_{DRONE}_vel_x.txt")
vy = np.loadtxt(f"{PATH}/out/drone_{DRONE}_vel_y.txt")
vz = np.loadtxt(f"{PATH}/out/drone_{DRONE}_vel_z.txt")
vyaw = np.loadtxt(f"{PATH}/out/drone_{DRONE}_vel_yaw.txt")

if err.size == 1:
    print("There's no data for this drone")
    sys.exit()

print(
    f"""
Drone {GEN_YAML['Leader_Follower']} with control ({CONTROLS[GEN_YAML['Leader_Follower']][DRONE_YAML['control']-1]})
Tiempo total -> {time[-1]:5f}
{f'Error final -> {err[-1]:5f} -- Max error -> {np.max(err[1:], axis=0):5f}' if len(err.shape) == 1 else f'Error final -> ({err[-1, 0]:5f}, {err[-1, 1]:5f}, {err[-1, 2]:5f}) -- Max error -> ({np.max(err[1:, 0], axis=0):5f}, {np.max(err[1:, 1], axis=0):5f}, {np.max(err[1:, 2], axis=0):5f})'}
Velocidad final -> ({vx[-1]:5f}, {vy[-1]:5f}, {vz[-1]:5f}, {vyaw[-1]:5f})
Promedio de tiempo por frame -> {np.mean(time[1:] - time[:-1]):5f}
"""
)

#########################  FIRST PLOT ERROR  #########################

ax[0].title.set_text(f"Error")
if len(err.shape) == 1:
    ax[0].step(
        time[1:],
        err[1:],
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
            time[1:],
            err[1:, i],
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
        time[1:],
        np.linalg.norm(err[1:, :], axis=1),
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
    ax[0].plot(time[1:], err_pix[1:], "-", c="r", label="Error (px)")
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
ax[1].step(time[1:], vx[1:], "-", label="$V_x$ (m/s)")
ax[1].step(time[1:], vy[1:], "-", label="$V_y$ (m/s)")
ax[1].step(time[1:], vz[1:], "-", label="$V_z$ (m/s)")
ax[1].step(time[1:], vyaw[1:], "-", label="$W_z$ (rad/s)")
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
                    time[1:],
                    lamb[1:],
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
                    time[1:],
                    np.mean(lamb[1:], axis=1),
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
        time[1:],
        integ[1:, 0] * Gains[0][1][1:, 0],
        label="$I_x * \lambda_{ki}$",
        alpha=0.7,
    )
    ax[3].plot(
        time[1:],
        integ[1:, 1] * Gains[0][1][1:, 1],
        label="$I_y * \lambda_{ki}$",
        alpha=0.7,
    )
    ax[3].plot(
        time[1:],
        integ[1:, 2] * Gains[0][1][1:, 2],
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
fig.tight_layout()
fig.savefig(f"{PATH}/out/out_velocities.png", bbox_inches="tight", pad_inches=0.1)

plt.show()
