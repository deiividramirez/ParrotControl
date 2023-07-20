import matplotlib.pyplot as plt
import numpy as np
import pathlib
import sys

PATH = pathlib.Path(__file__).parent.absolute()
colors = ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray", "olive"]

dron = sys.argv[1] if len(sys.argv) > 1 else 1

fig, ax = plt.subplots(
    3,
    1,
    figsize=(10, 5),
    sharex=True,
    num=f"Velocidades y errores para todos los drones",
)

err = np.loadtxt(f"{PATH}/out/drone_{dron}_error.txt")
err_pix = np.loadtxt(f"{PATH}/out/drone_{dron}_errorPix.txt")
time = np.loadtxt(f"{PATH}/out/drone_{dron}_time.txt")
vx = np.loadtxt(f"{PATH}/out/drone_{dron}_vel_x.txt")
vy = np.loadtxt(f"{PATH}/out/drone_{dron}_vel_y.txt")
vz = np.loadtxt(f"{PATH}/out/drone_{dron}_vel_z.txt")
vyaw = np.loadtxt(f"{PATH}/out/drone_{dron}_vel_yaw.txt")
# intx = np.loadtxt(f"{PATH}/out/drone_{dron}.txt")
# inty = np.loadtxt(f"{PATH}/out/drone_{dron}.txt")
# intz = np.loadtxt(f"{PATH}/out/drone_{dron}.txt")

if err.size == 1:
    print("There's no data for this drone")
    sys.exit()

print(
    f"""
Drone {dron}
Tiempo total -> {time[-1]}
Error final -> {err[-1]} -- Max error -> {np.max(err[1:], axis=0)}
Velocidad final -> {vx[-1], vy[-1], vz[-1], vyaw[-1]}
"""
)
ax[0].title.set_text(f"Drone {dron}")
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
            color=colors[i],
            label=f"Error (c) {'x' if i == 0 else ('y' if i == 1 else 'z')}",
            where="post",
        )
        ax[0].plot(
            [time[1], time[-1]],
            [err[-1, i], err[-1, i]],
            "--",
            color=colors[i],
            label=f"y={err[-1, i]:.3f}",
            alpha=0.5,
        )

if np.any(err_pix != 0):
    err_pix = err_pix
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
    [0, time[-1]],
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

#########################################################################################
ax[1].plot(
    [0, time[-1]],
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

#########################################################################################

for index_j, j in enumerate(["v", "w"]):
    for index_i, i in enumerate(["kp", "ki"]):
        try:
            lamb = np.loadtxt(f"{PATH}/out/drone_{dron}_{j}_{i}.txt")
            if np.any(lamb != 0):
                ax[2].plot(
                    time[1:],
                    lamb[1:],
                    label=j + ": $\lambda_{" + i + "}$",
                    color=colors[index_j * 3 + index_i],
                )
                ax[2].plot(
                    [time[1], time[-1]],
                    [lamb[-1], lamb[-1]],
                    "--",
                    color=colors[index_j * 3 + index_i],
                    label=f"y={lamb[-1]:3f}",
                    alpha=0.25,
                )
        except:
            pass
ax[2].legend(loc="center left", bbox_to_anchor=(1, 0.5))
ax[2].set_ylabel("Lambda")
ax[2].set_xlabel("Tiempo (s)")

# ax[3].plot(time[1:], intx[1:], label="$I_x$")
# ax[3].plot(time[1:], inty[1:], label="$I_y$")
# ax[3].plot(time[1:], intz[1:], label="$I_z$")
# ax[3].legend(loc="center left", bbox_to_anchor=(1, 0.5))
# ax[3].set_ylabel("Integrales")

fig.tight_layout()
fig.savefig(f"{PATH}/out/out_velocities.png", bbox_inches="tight", pad_inches=0.1)

plt.show()
