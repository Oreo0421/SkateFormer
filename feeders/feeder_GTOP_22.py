import numpy as np
import random

from torch.utils.data import Dataset
from feeders import tools


class Feeder(Dataset):
    """
    Mimic SkateFormer NTU feeder style, but for custom 22-joint 3D skeleton.

    Expected NPZ keys (same convention as NTU feeder):
      - x_train: (N, T, V*3)  OR (N, T, V, 3) OR (N, C, T, V, M)
      - y_train: (N, num_class) one-hot
      - x_test : same as x_train
      - y_test : same as y_train

    Output:
      data_numpy: (C, T, V, M)  (C=3 for xyz)
      index_t   : indices after crop/resize (from tools)
      label     : int class id
      index     : sample index
    """

    def __init__(self,
                 data_path,
                 label_path=None,
                 p_interval=1,
                 split='train',
                 data_type='j',
                 aug_method='z',
                 intra_p=0.5,
                 inter_p=0.0,
                 window_size=-1,
                 debug=False,
                 thres=64,
                 uniform=False,
                 partition=False,
                 # custom extras for your dataset
                 num_point=22,
                 num_dim=3,
                 num_people=1,
                 # normalize
                 root_center=True,
                 root_idx=0):

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.split = split
        self.data_type = data_type
        self.aug_method = aug_method
        self.intra_p = intra_p
        self.inter_p = inter_p
        self.window_size = window_size
        self.p_interval = p_interval
        self.thres = thres
        self.uniform = uniform
        self.partition = partition

        self.num_point = int(num_point)     # V
        self.num_dim = int(num_dim)         # xyz -> 3
        self.num_people = int(num_people)   # M -> 1

        self.root_center = bool(root_center)
        self.root_idx = int(root_idx)

        self.load_data()

        # NOTE:
        # NTU feeder uses hard-coded body-part reindex for V=25. :contentReference[oaicite:1]{index=1}
        # For your V=22, unless you have a specific partition order, keep partition=False.
        # If you REALLY need partition, implement your own new_idx here (e.g., from kin_parent).
        if self.partition:
            raise NotImplementedError(
                "partition=True is not implemented for V=22 in this feeder. "
                "Use partition=False, or tell me your desired joint groups/order and I’ll add it."
            )

    def load_data(self):
        npz_data = np.load(self.data_path, allow_pickle=True)

        if self.split == 'train':
            self.data = npz_data['x_train']
            self.label = np.where(npz_data['y_train'] > 0)[1]
            self.sample_name = ['train_' + str(i) for i in range(len(self.data))]
        elif self.split == 'test':
            self.data = npz_data['x_test']
            self.label = np.where(npz_data['y_test'] > 0)[1]
            self.sample_name = ['test_' + str(i) for i in range(len(self.data))]
        else:
            raise NotImplementedError('data split only supports train/test')

        self.data = np.array(self.data)

        # ---- Normalize data shape to (N, C, T, V, M) ----
        # Supported input shapes:
        # 1) (N, T, V*3)      -> reshape to (N, C=3, T, V, M=1)
        # 2) (N, T, V, 3)     -> transpose to (N, C=3, T, V, M=1)
        # 3) (N, C, T, V, M)  -> keep
        if self.data.ndim == 3:
            # (N, T, D)
            N, T, D = self.data.shape
            if D == self.num_point * self.num_dim:
                # (N,T,V*3) -> (N,T,V,3)
                self.data = self.data.reshape(N, T, self.num_point, self.num_dim)
                # -> (N,3,T,V,1)
                self.data = self.data.transpose(0, 3, 1, 2)[:, :, :, :, None]
            else:
                raise ValueError(
                    f"Unsupported x shape {self.data.shape}. "
                    f"Expected D=V*3={self.num_point*self.num_dim}."
                )

        elif self.data.ndim == 4:
            # (N, T, V, 3) or (N, T, 3, V) etc.
            N = self.data.shape[0]
            # assume (N,T,V,3)
            if self.data.shape[-1] == self.num_dim and self.data.shape[2] == self.num_point:
                # (N,T,V,3) -> (N,3,T,V,1)
                self.data = self.data.transpose(0, 3, 1, 2)[:, :, :, :, None]
            else:
                raise ValueError(
                    f"Unsupported 4D x shape {self.data.shape}. "
                    f"Expected (N,T,V,3) with V={self.num_point}."
                )

        elif self.data.ndim == 5:
            # (N, C, T, V, M)
            if self.data.shape[1] != self.num_dim or self.data.shape[3] != self.num_point:
                raise ValueError(
                    f"Unsupported 5D x shape {self.data.shape}. "
                    f"Expected (N,3,T,22,M)."
                )
        else:
            raise ValueError(f"Unsupported x ndim={self.data.ndim}, shape={self.data.shape}")

        # debug trim
        if self.debug:
            self.data = self.data[:min(200, len(self.data))]
            self.label = self.label[:min(200, len(self.label))]
            self.sample_name = self.sample_name[:min(200, len(self.sample_name))]

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def _root_center(self, data_numpy):
        # data_numpy: (C,T,V,M)
        if (not self.root_center) or data_numpy.shape[0] != 3:
            return data_numpy
        root = data_numpy[:, :, self.root_idx:self.root_idx + 1, :]  # (3,T,1,M)
        return data_numpy - root

    def __getitem__(self, index):
        data_numpy = self.data[index]  # (C,T,V,M)
        label = self.label[index]
        data_numpy = np.array(data_numpy)

        # valid frames: same logic as NTU feeder :contentReference[oaicite:2]{index=2}
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
        num_people = np.sum(data_numpy.sum(0).sum(0).sum(0) != 0)

        # crop/resize
        if self.uniform:
            data_numpy, index_t = tools.valid_crop_uniform(
                data_numpy, valid_frame_num, self.p_interval,
                self.window_size, self.thres
            )
        else:
            data_numpy, index_t = tools.valid_crop_resize(
                data_numpy, valid_frame_num, self.p_interval,
                self.window_size, self.thres
            )

        # root-center (recommended for action recognition)
        data_numpy = self._root_center(data_numpy)

        if self.split == 'train':
            # intra-instance augmentation (same structure as NTU) :contentReference[oaicite:3]{index=3}
            p = np.random.rand(1)

            if p < self.intra_p:

                if 'a' in self.aug_method:
                    if np.random.rand(1) < 0.5:
                        data_numpy = data_numpy[:, :, :, np.array([1, 0])]  # swap persons (if M=2)

                if 'b' in self.aug_method:
                    if num_people == 2 and data_numpy.shape[-1] == 2:
                        if np.random.rand(1) < 0.5:
                            axis_next = np.random.randint(0, 1)
                            temp = data_numpy.copy()
                            C, T, V, M = data_numpy.shape
                            x_new = np.zeros((C, T, V))
                            temp[:, :, :, axis_next] = x_new
                            data_numpy = temp

                # geometry/noise aug use the same tools
                if '1' in self.aug_method:
                    data_numpy = tools.shear(data_numpy, p=0.5)
                if '2' in self.aug_method:
                    data_numpy = tools.rotate(data_numpy, p=0.5)
                if '3' in self.aug_method:
                    data_numpy = tools.scale(data_numpy, p=0.5)
                if '4' in self.aug_method:
                    data_numpy = tools.spatial_flip(data_numpy, p=0.5)
                if '5' in self.aug_method:
                    data_numpy, index_t = tools.temporal_flip(data_numpy, index_t, p=0.5)
                if '6' in self.aug_method:
                    data_numpy = tools.gaussian_noise(data_numpy, p=0.5)
                if '7' in self.aug_method:
                    data_numpy = tools.gaussian_filter(data_numpy, p=0.5)
                if '8' in self.aug_method:
                    data_numpy = tools.drop_axis(data_numpy, p=0.5)
                if '9' in self.aug_method:
                    data_numpy = tools.drop_joint(data_numpy, p=0.5)

            # inter-instance augmentation (AdaIN bone length) - keep same style :contentReference[oaicite:4]{index=4}
            elif (p < (self.intra_p + self.inter_p)) and (p >= self.intra_p):
                adain_idx = random.choice(np.where(self.label == label)[0])
                data_adain = np.array(self.data[adain_idx])
                f_num = np.sum(data_adain.sum(0).sum(-1).sum(-1) != 0)
                # align time indices
                t_idx = np.round((index_t + 1) * f_num / 2).astype(int)
                t_idx = np.clip(t_idx, 0, data_adain.shape[1] - 1)
                data_adain = data_adain[:, t_idx]
                data_numpy = tools.skeleton_adain_bone_length(data_numpy, data_adain)
            else:
                data_numpy = data_numpy.copy()

        # modality (same as NTU) :contentReference[oaicite:5]{index=5}
        if self.data_type == 'b':
            j2b = tools.joint2bone()
            data_numpy = j2b(data_numpy)
        elif self.data_type == 'jm':
            data_numpy = tools.to_motion(data_numpy)
        elif self.data_type == 'bm':
            j2b = tools.joint2bone()
            data_numpy = j2b(data_numpy)
            data_numpy = tools.to_motion(data_numpy)
        else:
            data_numpy = data_numpy.copy()

        # partition disabled for now
        return data_numpy, index_t, label, index

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod
