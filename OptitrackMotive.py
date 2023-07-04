# import natnetclient as natnet
# client = natnet.NatClient(client_ip='127.0.0.1', data_port=1511, comm_port=1510)

import natnet
client = natnet.Client.connect(server="10.10.101.28")
client.set_callback(
    lambda rigid_bodies, markers, timing: print(rigid_bodies))
client.spin()
