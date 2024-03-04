import pandas as pd
import numpy as np
import myosuite
import torch
import gym
import bids

def get_activations(model, x):
    '''
    Parameters
    ----------
    model : mjrl.utils.fc_network.FCNetwork
    x : a ([n_observations,] observation_size) np.array
        An observation vector in same form as returned by env.step().
        Observations are of shape (jnt_pos + jnt_vel + last_action + pos_error,)

    Returns
    ---------
    activations : list of ([n_observations], layer_width) np.arrays
        The activations in each layer of `model` when `x` passed through.

    Notes
    ---------
    `model` can be obtained from a myosuite NPG baseline model `pi` which
    has been loaded with pickle with `model = pi.model`.
    '''
    activations = []
    out = torch.from_numpy(x).float()
    out = (out - model.in_shift) / (model.in_scale + 1e-8)
    activations.append(out.detach().numpy()) # input layer
    for i in range(len(model.fc_layers)):
      out = model.fc_layers[i](out)
      if i != len(model.fc_layers) - 1: # i.e. not output layer
          out = model.nonlinearity(out)
      activations.append(out.detach().numpy().astype(float))
    return activations

def get_joints_and_muscles(env_name = 'myoHandPoseRandom-v0'):
    '''
    Parameters
    -----------
    env_name : str
        The name of a myosuite environment

    Returns
    ------------
    joints : list of str
        The ordered names of the joints in environment's model
    muscles : list of str
        The ordered names of the muscles in env's model
    '''
    env = gym.make(env_name)
    mjcModel = env.env.sim.model
    muscles = [mjcModel.actuator(i).name for i in range(mjcModel.na)]
    joints = [mjcModel.joint(i).name for i in range(mjcModel.njnt)]
    return joints, muscles


def _calc_error(joint_angles, events):
    '''
    Calculates signed angular distance from target joint positions,
    assuming subjects have reached target by last half second
    of each trial.
    '''
    events = events.copy()
    events['offset'] = events.onset + events.duration
    pos_error = pd.DataFrame(
        np.full_like(joint_angles, np.nan),
        columns = [col + '_err' for col in joint_angles.columns]
    ).set_index(joint_angles.index)
    for i, (onset, offset) in enumerate(zip(events.onset, events.offset)):
        # we assume they've reached target by last 1/2 second
        target = joint_angles[offset - .5 : offset].mean(0)
        if i == 0:
            pos_error[:offset] = target - joint_angles[:offset]
        else:
            pos_error[onset:] = target - joint_angles[onset:]
    return pos_error


def _calc_vel(joint_angles):
    '''
    computes angular velocities of joint angles
    '''
    t = joint_angles.index
    joint_grad = np.gradient(joint_angles.to_numpy(), axis = 0)
    joint_vel = joint_grad * np.gradient(t)[:, np.newaxis]
    joint_vel = pd.DataFrame(
        joint_vel,
        columns = [col + '_vel' for col in joint_angles.columns]
    ).set_index(t)
    return joint_vel

def _fill_in_last_action(joint_angles, muscles):
    '''
    creates a dataframe of zeros with columns for each muscle
    '''
    n_muscles = len(muscles)
    n_times = joint_angles.shape[0]
    actions = pd.DataFrame(
        np.zeros((n_times, n_muscles)),
        columns = [m + '_act' for m in muscles]
    ).set_index(joint_angles.index)
    return actions


def joints_to_myo_obs(joint_angles, times, events, env_name):
    '''
    Parameters
    ----------
    joint_angles : an (n_observations, n_joints_in_rec) pd.DataFrame
        Contains joint angle measurements, with joint names
        in myosuite nomenclature as column names.
    times : an (n_observations,) np.array
        timestamps for each joint angle measurement
    events : pd.DataFrame
        A BIDS events dataframe indicating when the subject was given
        a new joint pose target.
    env_name : str
        The name of a myosuite environment

    Returns
    ----------
    obs : an (n_observations, obs_size) pd.DataFrame
        Joint measurements in format expected by myosuite environment,
        i.e. columns in order of:
            - position (unmeasured angles are set to zero)
            - angular velocity
            - last action (set to zero here, since not measured)
            - position error
    Notes
    -------
    You can find the joint names for a myosuite environment using
    the `get_joints_and_muscles` function in this module.
    '''
    joints, muscles = get_joints_and_muscles(env_name)
    joint_angles = joint_angles.copy()
    joint_angles = joint_angles.set_index(times) # needed for helper functions

    ## fill in missing joints
    try:
        assert([j for j in joints if j in joint_angles.columns])
    except:
        msg = 'No column names in joint_angles match joint names %s.'%env_name
        msg += ' Acceptable joint names are %s.'%', '.join(joints)
        raise Exception(msg)
    missing = [j for j in joints if not j in joint_angles.columns]
    print('\nFilling in missing joints:', missing)
    for jnt in missing: # fill in zeros for any missing joints
        joint_angles[jnt] = 0
    pos = joint_angles[joints] # and reorder to match myosuite

    ## compute derivative quantities in myosuite observation
    vel = _calc_vel(pos)
    act = _fill_in_last_action(pos, muscles)
    pos_error = _calc_error(pos, events)

    pos = pos.rename(columns = {
        col: col + '_pos' for col in pos.columns
    })
    obs = pd.concat(
        [pos, vel, act, pos_error],
        axis = 1
    )
    return obs
