""" Testing functions for the recreate_LFP module. """
import numpy as np

def assert_established_network_pos_rot(df, network):
    """ Function designed to evaluate built networks by comparing cell positions
    and rotations with a Pandas DataFrame of the correct data.
    ----------
    Parameters
        df : Pandas DataFrame object
            Dataframe of the cell positions and rotations which should be cross-
            checked with the network generated.
            Columns should be: [gid, x, y, z, x_rot, y_rot, z_rot]
        network : LFPy.network object
            Generated LFPy network of cells.
    """
    errors = 0
    pos_ = [[None] * 3 for i in range(2)]
    rot_ = [[None] * 3 for i in range(2)]
    for name in network.population_names:
        for no, cell in enumerate(network.populations[name].cells):
            """ Loop over all the cells in the network and assert that they
            have the same position and rotation as the DF listing. """
            pos_[0][0] = cell.somapos[0]
            pos_[0][1] = cell.somapos[1]
            pos_[0][2] = cell.somapos[2]
            pos_[1][0] = df.loc[df['gid']==cell.gid, 'x'].item()
            pos_[1][1] = df.loc[df['gid']==cell.gid, 'y'].item()
            pos_[1][2] = df.loc[df['gid']==cell.gid, 'z'].item()

            # The first two are predefined in the current state of things
            rot_[0][0] = np.pi/2.
            rot_[0][1] = 0
            rot_[0][2] = network.populations[name].rotations[no]
            rot_[1][0] = df.loc[df['gid']==cell.gid, 'x_rot'].item()
            rot_[1][1] = df.loc[df['gid']==cell.gid, 'y_rot'].item()
            rot_[1][2] = df.loc[df['gid']==cell.gid, 'z_rot'].item()

            errors += sum([i != j for i,j in zip(pos_[0],pos_[1])])
            errors += sum([i != j for i,j in zip(rot_[0],rot_[1])])

    msg = "Network population not established correctly. Cell positions and/or rotations do not match."
    if errors != 0:
        print("Usage Warning: " + msg)

def assert_established_network_syn_out(spike_trains, network):
    """ Create testing functions to assess whether the synaptic inputs are set
    up properly. This one focuses on the outside stimulus synapses. The
    spike_trains list has the format: len(spike_trains) = numCells. The next
    layer contains all the synapses for that cell (len(spike_trains[0])=nidx).
    The first element in this list in the list spike_trains[0][0] is the cell
    idx that synapse is connected to and the others are the spike times. """
    errors = 0
    for name in network.population_names:
        for cell in network.populations[name].cells:
            for ctr, _ in enumerate(cell.synapses):
                real_list = cell.sptimeslist[ctr]
                fake_list = spike_trains[cell.gid][ctr][1:]

                real_idx = cell.synapses[ctr].idx
                fake_idx = spike_trains[cell.gid][ctr][0]

                # Evaluate the lists and indices to be equal.
                errors += sum([i != j for i,j in zip(real_list,fake_list)])
                if real_idx != fake_idx:
                    errors += 1

    msg = "Outside stimulus of network not established correctly. Outside synaptic spike trains mismatch."
    if errors != 0:
        print("Usage Warning: " + msg)

def assert_established_network_syn_net(df, network, nidx):
    """ Create testing functions to assess whether the synaptic inputs are set
    up properly. This one focuses on the network connection synapses. The
    columns of the dataframe should be: [gid, idx, weight, delay, pre_gid].
    The parameter 'nidx' needs to be parsed, as the connection synapses are
    the ones which are listed after the outside stimulus ones. """

    """ Only current (LFP version) accessible synapse information to be found
    is contained in the cell.synapses.kwargs object, listing only weights. """

    # Compare the weights of the synapses systematically to assess equality
    errors = 0
    for name in network.population_names:
        for cell in network.populations[name].cells:
            real_connList = cell.synapses[nidx:]
            for ctr, (_, row) in enumerate(df[df['gid']==cell.gid].iterrows()):
                # print(row)  # idx, weight, delay, pre_gid
                real_weight = real_connList[ctr].kwargs['weight']
                fake_weight = row['weight'].item()

                if real_weight != fake_weight:
                    errors += 1

    msg = "Network connectivity not reproduced correctly. Inside synaptic connection parameters mismatch."
    if errors != 0:
        print("Usage Warning: " + msg)
