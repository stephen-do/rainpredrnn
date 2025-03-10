import torch
import os


def save_model(network, epoch, configs):
    stats = {}
    stats['net_param'] = network.state_dict()
    checkpoint_path = os.path.join(configs.save_dir, configs.model_name, 'model' + '-' + str(epoch) + '.ckpt')
    torch.save(stats, checkpoint_path)
    print("saved model to %s" % checkpoint_path)


def load_model(network, checkpoint_path):
    stats = torch.load(checkpoint_path, map_location='cuda')
    network.load_state_dict(stats['net_param'])
    print('loaded model: %s successfully' % checkpoint_path)
    return network
