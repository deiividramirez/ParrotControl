from pyparrot.Bebop import Bebop

bebop = Bebop(drone_type="Bebop2")

print("connecting")
success = bebop.connect(10)
print(success)

print("sleeping")
bebop.smart_sleep(5)

# bebop.start_video_stream()
bebop.ask_for_state_update()

print("sleeping")
bebop.smart_sleep(5)

bebop.safe_land(10)