import os
import os.path
import sys
import torch
import torch.utils.data as data
import numpy as np
import scipy.spatial as spatial
import trimesh

from source.base import mesh_io
from source.base import utils
from source.base import file_utils
from source import sdf


def load_shape(point_filename, imp_surf_query_filename, imp_surf_dist_filename,normals_file_name,
               query_grid_resolution=None, epsilon=None):
    """
    do NOT modify the returned points! kdtree uses a reference, not a copy of these points,
    so modifying the points would make the kdtree give incorrect results
    :param point_filename:
    :param imp_surf_query_filename:
    :param imp_surf_dist_filename:
    :param normals_file_name:
    :param query_grid_resolution: if not None: create query points at grid of this resolution
    :param epsilon:
    :return:
    """

    mmap_mode = None
    # mmap_mode = 'r'

    pts_np = np.load(point_filename + '.npy', mmap_mode=mmap_mode)
    if pts_np.shape[1] > 3:
        pts_np = pts_np[:, 0:3]
    if pts_np.dtype != np.float32:
        print('Warning: pts_np must be converted to float32: {}'.format(point_filename))
        pts_np = pts_np.astype(np.float32)

    # otherwise KDTree construction may run out of recursions
    leaf_size = 1000
    sys.setrecursionlimit(int(max(1000, round(pts_np.shape[0]/leaf_size))))
    kdtree = spatial.cKDTree(pts_np, leaf_size)

    if imp_surf_dist_filename is not None:
        imp_surf_dist_ms = np.load(imp_surf_dist_filename, mmap_mode=mmap_mode)
        if imp_surf_dist_ms.dtype != np.float32:
            print('Warning: imp_surf_dist_ms must be converted to float32')
            imp_surf_dist_ms = imp_surf_dist_ms.astype(np.float32)
    else:
        imp_surf_dist_ms = None

    if imp_surf_query_filename is not None:
        imp_surf_query_point_ms = np.load(imp_surf_query_filename, mmap_mode=mmap_mode)
        if imp_surf_query_point_ms.dtype != np.float32:
            print('Warning: imp_surf_query_point_ms must be converted to float32')
            imp_surf_query_point_ms = imp_surf_query_point_ms.astype(np.float32)
    elif query_grid_resolution is not None:
        # get query points near at grid nodes near the point cloud
        # get patches for those query points
        imp_surf_query_point_ms = sdf.get_voxel_centers_grid_smaller_pc(
            pts=pts_np, grid_resolution=query_grid_resolution,
            distance_threshold_vs=epsilon)
    else:
        #imp_surf_query_point_ms = None
        # wsy:radom sample 1000 quert points on surface and get their normals
        sub_sample_ids = np.random.randint(low=0, high=pts_np.shape[0], size=1000)
        imp_surf_query_point_ms=pts_np[sub_sample_ids, :]
        if imp_surf_query_point_ms.dtype != np.float32:
            print('Warning: imp_surf_query_point_ms must be converted to float32')
            imp_surf_query_point_ms = imp_surf_query_point_ms.astype(np.float32)
        normals = np.loadtxt(normals_file_name).astype('float32')
        #normals=np.load(normals_file_name, mmap_mode=mmap_mode)
        normals=normals[sub_sample_ids, :]
        if normals.dtype != np.float32:
            print('Warning: normals must be converted to float32')
            normals = normals.astype(np.float32)




    return Shape(
        pts=pts_np, kdtree=kdtree,
        imp_surf_query_point_ms=imp_surf_query_point_ms, imp_surf_dist_ms=imp_surf_dist_ms,query_points_normals=normals)


class SequentialPointcloudPatchSampler(data.sampler.Sampler):

    def __init__(self, data_source):
        self.data_source = data_source
        self.total_patch_count = None

        self.total_patch_count = 0
        for shape_ind, _ in enumerate(self.data_source.shape_names):
            self.total_patch_count = self.total_patch_count + self.data_source.shape_patch_count[shape_ind]

    def __iter__(self):
        return iter(range(self.total_patch_count))

    def __len__(self):
        return self.total_patch_count


class SequentialShapeRandomPointcloudPatchSampler(data.sampler.Sampler):

    def __init__(self, data_source, patches_per_shape, seed=None, sequential_shapes=False, identical_epochs=False):
        self.data_source = data_source
        self.patches_per_shape = patches_per_shape
        self.sequential_shapes = sequential_shapes
        self.seed = seed
        self.identical_epochs = identical_epochs
        self.total_patch_count = None
        self.shape_patch_inds = None

        if self.seed is None:
            self.seed = np.random.random_integers(0, 2**32-1, 1)[0]
        self.rng = np.random.RandomState(self.seed)

        self.total_patch_count = 0
        for shape_ind, _ in enumerate(self.data_source.shape_names):
            self.total_patch_count = self.total_patch_count + \
                                     min(self.patches_per_shape, self.data_source.shape_patch_count[shape_ind])

    def __iter__(self):

        # optionally always pick the same permutation (mainly for debugging)
        if self.identical_epochs:
            self.rng.seed(self.seed)

        # global point index offset for each shape
        shape_patch_offset = list(np.cumsum(self.data_source.shape_patch_count))
        shape_patch_offset.insert(0, 0)
        shape_patch_offset.pop()

        shape_inds = range(len(self.data_source.shape_names))

        if not self.sequential_shapes:
            shape_inds = self.rng.permutation(shape_inds)

        # return a permutation of the points in the dataset
        # where all points in the same shape are adjacent (for performance reasons):
        # first permute shapes, then concatenate a list of permuted points in each shape
        self.shape_patch_inds = [[]]*len(self.data_source.shape_names)
        point_permutation = []
        for shape_ind in shape_inds:
            start = shape_patch_offset[shape_ind]
            end = shape_patch_offset[shape_ind]+self.data_source.shape_patch_count[shape_ind]

            global_patch_inds = self.rng.choice(range(start, end),
                                                size=min(self.patches_per_shape, end-start), replace=False)
            point_permutation.extend(global_patch_inds)

            # save indices of shape point subset
            self.shape_patch_inds[shape_ind] = global_patch_inds - start

        return iter(point_permutation)

    def __len__(self):
        return self.total_patch_count


class RandomPointcloudPatchSampler(data.sampler.Sampler):

    def __init__(self, data_source, patches_per_shape, seed=None, identical_epochs=False):
        self.data_source = data_source
        self.patches_per_shape = patches_per_shape
        self.seed = seed
        self.identical_epochs = identical_epochs
        self.total_patch_count = None

        if self.seed is None:
            self.seed = np.random.random_integers(0, 2**32-1, 1)[0]
        self.rng = np.random.RandomState(self.seed)

        self.total_patch_count = 0
        for shape_ind, _ in enumerate(self.data_source.shape_names):
            self.total_patch_count = self.total_patch_count + \
                                     min(self.patches_per_shape, self.data_source.shape_patch_count[shape_ind])

    def __iter__(self):

        # optionally always pick the same permutation (mainly for debugging)
        if self.identical_epochs:
            self.rng.seed(self.seed)

        return iter(self.rng.choice(
            sum(self.data_source.shape_patch_count), size=self.total_patch_count, replace=False))

    def __len__(self):
        return self.total_patch_count


class Shape:
    def __init__(self, pts, kdtree,
                 imp_surf_query_point_ms, imp_surf_dist_ms,query_points_normals):
        self.pts = pts
        self.kdtree = kdtree
        self.imp_surf_query_point_ms = imp_surf_query_point_ms
        self.imp_surf_dist_ms = imp_surf_dist_ms
        #wsy: add normals
        self.query_points_normals=query_points_normals


class Cache:
    def __init__(self, capacity, loader, loadfunc):
        self.elements = {}
        self.used_at = {}
        self.capacity = capacity
        self.loader = loader
        self.loadfunc = loadfunc
        self.counter = 0

    def get(self, element_id):
        if element_id not in self.elements:
            # cache miss

            # if at capacity, throw out least recently used item
            if len(self.elements) >= self.capacity:
                remove_id = min(self.used_at, key=self.used_at.get)
                del self.elements[remove_id]
                del self.used_at[remove_id]

            # load element
            self.elements[element_id] = self.loadfunc(self.loader, element_id)

        self.used_at[element_id] = self.counter
        self.counter += 1

        return self.elements[element_id]


class PointcloudPatchDataset(data.Dataset):

    def __init__(self, root, shape_list_filename, points_per_patch, patch_radius, patch_features, epsilon,
                 seed=None, identical_epochs=False, center='point',
                 cache_capacity=1, point_count_std=0.0,
                 pre_processed_patches=False, query_grid_resolution=None,
                 sub_sample_size=500, reconstruction=False, uniform_subsample=False,
                 num_workers=1):

        # initialize parameters
        self.root = root
        self.shape_list_filename = shape_list_filename
        self.patch_features = patch_features
        self.points_per_patch = points_per_patch
        self.patch_radius = patch_radius
        self.identical_epochs = identical_epochs
        self.pre_processed_patches = pre_processed_patches
        self.center = center
        self.point_count_std = point_count_std
        self.seed = seed
        self.query_grid_resolution = query_grid_resolution
        self.sub_sample_size = sub_sample_size
        self.reconstruction = reconstruction
        self.num_workers = num_workers
        self.epsilon = epsilon
        self.uniform_subsample = uniform_subsample

        self.include_connectivity = False
        self.include_imp_surf = False
        self.include_p_index = False
        self.include_patch_pts_ids = False
        #wsy: add normals
        self.include_normals=False
        for pfeat in self.patch_features:
            if pfeat == 'imp_surf':
                self.include_imp_surf = True
            elif pfeat == 'imp_surf_magnitude':
                self.include_imp_surf = True
            elif pfeat == 'imp_surf_sign':
                self.include_imp_surf = True
            elif pfeat == 'p_index':
                self.include_p_index = True
            elif pfeat == 'patch_pts_ids':
                self.include_patch_pts_ids = True
            #wsy: add normals
            elif pfeat == 'normals':
                self.include_normals = True
            else:
                raise ValueError('Unknown patch feature: %s' % pfeat)

        self.shape_cache = Cache(cache_capacity, self, PointcloudPatchDataset.load_shape_by_index)

        # get all shape names in the dataset
        self.shape_names = []
        with open(os.path.join(root, self.shape_list_filename)) as f:
            self.shape_names = f.readlines()
        self.shape_names = [x.strip() for x in self.shape_names]
        self.shape_names = list(filter(None, self.shape_names))

        # initialize rng for picking points in a patch
        if self.seed is None:
            self.seed = np.random.random_integers(0, 2**32-1, 1)[0]
        self.rng = np.random.RandomState(self.seed)

        # get basic information for each shape in the dataset
        self.shape_patch_count = []
        print('getting information for {} shapes'.format(len(self.shape_names)))
        for shape_ind, shape_name in enumerate(self.shape_names):
            # print('getting information for shape %s' % shape_name)

            def load_pts():
                # load from text file and save in more efficient numpy format
                #point_filename = os.path.join(self.root, '04_pts', shape_name + '.xyz')
                # wsy: modify point file name
                point_filename = os.path.join(self.root, shape_name + '.xyz')
                if os.path.isfile(point_filename) or os.path.isfile(point_filename + '.npy'):
                    pts = file_utils.load_npy_if_valid(point_filename, 'float32', mmap_mode='r')
                    if pts.shape[1] > 3:
                        pts = pts[:, 0:3]
                else:  # if no .xyz file, try .off and discard connectivity
                    mesh_filename = os.path.join(self.root, '04_pts', shape_name+'.xyz')
                    pts, _ = mesh_io.load_mesh(mesh_filename)
                    np.savetxt(fname=os.path.join(self.root, '04_pts', shape_name+'.xyz'), X=pts)
                    np.save(os.path.join(self.root, '04_pts', shape_name+'.xyz.npy'), pts)
                return pts

            if self.include_imp_surf:
                if self.reconstruction:
                    # get number of grid points near the point cloud
                    pts = load_pts()
                    grid_pts_near_surf_ms = \
                        sdf.get_voxel_centers_grid_smaller_pc(
                            pts=pts, grid_resolution=query_grid_resolution, distance_threshold_vs=self.epsilon)
                    self.shape_patch_count.append(grid_pts_near_surf_ms.shape[0])

                    # un-comment to get a debug output for the necessary query points
                    # mesh_io.write_off('debug/{}'.format(shape_name + '.off'), grid_pts_near_surf_ms, [])
                    # self.shape_patch_count.append(query_grid_resolution ** 3)  # full grid
                else:
                    query_dist_filename = os.path.join(self.root, '05_query_pts', shape_name + '.ply.npy')
                    query_dist = np.load(query_dist_filename)
                    self.shape_patch_count.append(query_dist.shape[0])
            else:
                pts = load_pts()
                #wsy: every shape include 1000 query points
                self.shape_patch_count.append(1000)

    # returns a patch centered at the point with the given global index
    # and the ground truth normal the the patch center
    def __getitem__(self, index):

        # find shape that contains the point with given global index
        shape_ind, patch_ind = self.shape_index(index)
        def get_patch_points(shape, query_point):

            from source.base import point_cloud

            # optionally always pick the same points for a given patch index (mainly for debugging)
            if self.identical_epochs:
                self.rng.seed((self.seed + index) % (2**32))

            patch_pts_ids = point_cloud.get_patch_kdtree(
                kdtree=shape.kdtree, rng=self.rng, query_point=query_point,
                patch_radius=self.patch_radius,
                points_per_patch=self.points_per_patch, n_jobs=1)
            # find -1 ids for padding
            patch_pts_pad_ids = patch_pts_ids == -1
            patch_pts_ids[patch_pts_pad_ids] = 0
            pts_patch_ms = shape.pts[patch_pts_ids, :]
            # replace padding points with query point so that they appear in the patch origin\
            pts_patch_ms[patch_pts_pad_ids, :] = query_point
            patch_radius_ms = utils.get_patch_radii(pts_patch_ms, query_point)\
                if self.patch_radius <= 0.0 else self.patch_radius
            pts_patch_ps = utils.model_space_to_patch_space(
                pts_to_convert_ms=pts_patch_ms, pts_patch_center_ms=query_point,
                patch_radius_ms=patch_radius_ms)

            return patch_pts_ids, pts_patch_ps, pts_patch_ms, patch_radius_ms

        shape = self.shape_cache.get(shape_ind)
        imp_surf_query_point_ms = shape.imp_surf_query_point_ms[patch_ind]
        # get neighboring points
        patch_pts_ids, patch_pts_ps, pts_patch_ms, patch_radius_ms = \
            get_patch_points(shape=shape, query_point=imp_surf_query_point_ms)
        imp_surf_query_point_ps = utils.model_space_to_patch_space_single_point(
            imp_surf_query_point_ms, imp_surf_query_point_ms, patch_radius_ms)

        # surf dist can be None because we have no ground truth for evaluation
        # need a number or Pytorch will complain when assembling the batch
        if self.reconstruction:
            imp_surf_dist_ms = np.array([np.inf])
            imp_surf_dist_sign_ms = np.array([np.inf])
        else:
            #imp_surf_dist_ms = shape.imp_surf_dist_ms[patch_ind]
            #imp_surf_dist_sign_ms = np.sign(imp_surf_dist_ms)
            #imp_surf_dist_sign_ms = 0.0 if imp_surf_dist_sign_ms < 0.0 else 1.0
            # wsy: set SDF is None
            imp_surf_dist_ms = None
            imp_surf_dist_sign_ms = None
            query_points_normals=shape.query_points_normals[patch_ind]

        if self.sub_sample_size > 0:
            pts_sub_sample_ms = utils.get_point_cloud_sub_sample(
                sub_sample_size=self.sub_sample_size, pts_ms=shape.pts,
                query_point_ms=imp_surf_query_point_ms, uniform=self.uniform_subsample)
        else:
            pts_sub_sample_ms = np.array([], dtype=np.float32)

        if not self.reconstruction:
            import trimesh.transformations as trafo
            # random rotation of shape and patch as data augmentation
            rand_rot = trimesh.transformations.random_rotation_matrix(self.rng.rand(3))
            # rand_rot = trimesh.transformations.identity_matrix()
            pts_sub_sample_ms = \
                trafo.transform_points(pts_sub_sample_ms, rand_rot).astype(np.float32)
            patch_pts_ps = \
                trafo.transform_points(patch_pts_ps, rand_rot).astype(np.float32)
            imp_surf_query_point_ms = \
                trafo.transform_points(np.expand_dims(imp_surf_query_point_ms, 0), rand_rot)[0].astype(np.float32)
            imp_surf_query_point_ps = \
                trafo.transform_points(np.expand_dims(imp_surf_query_point_ps, 0), rand_rot)[0].astype(np.float32)
            query_points_normals=\
                trafo.transform_points(np.expand_dims(query_points_normals, 0), rand_rot)[0].astype(np.float32)
        patch_data = dict()
        # create new arrays to close the memory mapped files
        patch_data['patch_pts_ps'] = patch_pts_ps
        patch_data['patch_radius_ms'] = np.array(patch_radius_ms, dtype=np.float32)
        patch_data['pts_sub_sample_ms'] = pts_sub_sample_ms
        patch_data['imp_surf_query_point_ms'] = imp_surf_query_point_ms
        patch_data['imp_surf_query_point_ps'] = np.array(imp_surf_query_point_ps)
        patch_data['imp_surf_ms'] = np.array([imp_surf_dist_ms], dtype=np.float32)
        #patch_data['imp_surf_magnitude_ms'] = np.array([np.abs(imp_surf_dist_ms)], dtype=np.float32)
        patch_data['imp_surf_dist_sign_ms'] = np.array([imp_surf_dist_sign_ms], dtype=np.float32)
        patch_data['normals']=query_points_normals
        # un-comment to get a debug output of a training sample
        # import evaluation
        # evaluation.visualize_patch(
        #     patch_pts_ps=patch_data['patch_pts_ps'], patch_pts_ms=pts_patch_ms,
        #     query_point_ps=patch_data['imp_surf_query_point_ps'],
        #     pts_sub_sample_ms=patch_data['pts_sub_sample_ms'], query_point_ms=patch_data['imp_surf_query_point_ms'],
        #     file_path='debug/patch_local_and_global.ply')
        # patch_sphere = trimesh.primitives.Sphere(radius=self.patch_radius, center=imp_surf_query_point_ms)
        # patch_sphere.export(file_obj='debug/patch_sphere.ply')
        # print('Debug patch outputs with radius {} in "debug" dir'.format(self.patch_radius))

        # convert to tensors
        for key in patch_data.keys():
            patch_data[key] = torch.from_numpy(patch_data[key])

        return patch_data

    def __len__(self):
        return sum(self.shape_patch_count)

    # translate global (dataset-wide) point index to shape index & local (shape-wide) point index
    def shape_index(self, index):
        shape_patch_offset = 0
        shape_ind = None
        shape_patch_ind = -1
        for shape_ind, shape_patch_count in enumerate(self.shape_patch_count):
            if index >= shape_patch_offset and index < (shape_patch_offset + shape_patch_count):
                shape_patch_ind = index - shape_patch_offset
                break
            shape_patch_offset = shape_patch_offset + shape_patch_count

        return shape_ind, shape_patch_ind

    # load shape from a given shape index
    def load_shape_by_index(self, shape_ind):
        # wsy:modify file name
        point_filename = os.path.join(self.root, self.shape_names[shape_ind]+'.xyz')
        normals_filename= os.path.join(self.root, self.shape_names[shape_ind]+'.normals')
        return load_shape(
            point_filename=point_filename,
            imp_surf_query_filename=None,
            imp_surf_dist_filename=None,
            query_grid_resolution=self.query_grid_resolution,
            normals_file_name=normals_filename,
            epsilon=self.epsilon,
            )

class our_PointcloudPatchDataset(data.Dataset):

    # patch radius as fraction of the bounding box diagonal of a shape
    def __init__(self, root, shape_list_filename, patch_radius, points_per_patch, patch_features,
                 seed=None, identical_epochs=False, use_pca=True, center='point', point_tuple=1, cache_capacity=1, point_count_std=0.0, sparse_patches=False):

        # initialize parameters
        self.root = root
        self.shape_list_filename = shape_list_filename
        self.patch_features = patch_features
        self.patch_radius = patch_radius
        self.points_per_patch = points_per_patch
        self.identical_epochs = identical_epochs
        self.use_pca = use_pca
        self.sparse_patches = sparse_patches
        self.center = center
        self.point_tuple = point_tuple
        self.point_count_std = point_count_std
        self.seed = seed

        self.include_normals = False
        self.include_curvatures = False
        for pfeat in self.patch_features:
            if pfeat == 'normal':
                self.include_normals = True
            elif pfeat == 'max_curvature' or pfeat == 'min_curvature':
                self.include_curvatures = True
            else:
                raise ValueError('Unknown patch feature: %s' % (pfeat))

        # self.loaded_shape = None
        self.load_iteration = 0
        self.shape_cache = Cache(cache_capacity, self, PointcloudPatchDataset.load_shape_by_index)

        # get all shape names in the dataset
        self.shape_names = []
        with open(os.path.join(root, self.shape_list_filename)) as f:
            self.shape_names = f.readlines()
        self.shape_names = [x.strip() for x in self.shape_names]
        self.shape_names = list(filter(None, self.shape_names))

        # initialize rng for picking points in a patch
        if self.seed is None:
            self.seed = np.random.random_integers(0, 2**32-1, 1)[0]
        self.rng = np.random.RandomState(self.seed)

        # get basic information for each shape in the dataset
        self.shape_patch_count = []
        self.patch_radius_absolute = []
        for shape_ind, shape_name in enumerate(self.shape_names):
            print('getting information for shape %s' % (shape_name))

            # load from text file and save in more efficient numpy format
            point_filename = os.path.join(self.root, shape_name+'.xyz')
            pts = np.loadtxt(point_filename).astype('float32')
            np.save(point_filename+'.npy', pts)

            if self.include_normals:
                normals_filename = os.path.join(self.root, shape_name+'.normals')
                normals = np.loadtxt(normals_filename).astype('float32')
                np.save(normals_filename+'.npy', normals)
            else:
                normals_filename = None

            if self.include_curvatures:
                curv_filename = os.path.join(self.root, shape_name+'.curv')
                curvatures = np.loadtxt(curv_filename).astype('float32')
                np.save(curv_filename+'.npy', curvatures)
            else:
                curv_filename = None

            if self.sparse_patches:
                pidx_filename = os.path.join(self.root, shape_name+'.pidx')
                patch_indices = np.loadtxt(pidx_filename).astype('int')
                np.save(pidx_filename+'.npy', patch_indices)
            else:
                pidx_filename = None

            shape = self.shape_cache.get(shape_ind)

            if shape.pidx is None:
                self.shape_patch_count.append(shape.pts.shape[0])
            else:
                self.shape_patch_count.append(len(shape.pidx))

            bbdiag = float(np.linalg.norm(shape.pts.max(0) - shape.pts.min(0), 2))
            self.patch_radius_absolute.append([bbdiag * rad for rad in self.patch_radius])

    # returns a patch centered at the point with the given global index
    # and the ground truth normal the the patch center
    def __getitem__(self, index):

        # find shape that contains the point with given global index
        shape_ind, patch_ind = self.shape_index(index)

        shape = self.shape_cache.get(shape_ind)
        if shape.pidx is None:
            center_point_ind = patch_ind
        else:
            center_point_ind = shape.pidx[patch_ind]

        # get neighboring points (within euclidean distance patch_radius)
        patch_pts = torch.zeros(self.points_per_patch*len(self.patch_radius_absolute[shape_ind]), 3, dtype=torch.float)
        # patch_pts_valid = torch.ByteTensor(self.points_per_patch*len(self.patch_radius_absolute[shape_ind])).zero_()
        patch_pts_valid = []
        scale_ind_range = np.zeros([len(self.patch_radius_absolute[shape_ind]), 2], dtype='int')
        for s, rad in enumerate(self.patch_radius_absolute[shape_ind]):
            patch_point_inds = np.array(shape.kdtree.query_ball_point(shape.pts[center_point_ind, :], rad))

            # optionally always pick the same points for a given patch index (mainly for debugging)
            if self.identical_epochs:
                self.rng.seed((self.seed + index) % (2**32))

            point_count = min(self.points_per_patch, len(patch_point_inds))

            # randomly decrease the number of points to get patches with different point densities
            if self.point_count_std > 0:
                point_count = max(5, round(point_count * self.rng.uniform(1.0-self.point_count_std*2)))
                point_count = min(point_count, len(patch_point_inds))

            # if there are too many neighbors, pick a random subset
            if point_count < len(patch_point_inds):
                patch_point_inds = patch_point_inds[self.rng.choice(len(patch_point_inds), point_count, replace=False)]

            start = s*self.points_per_patch
            end = start+point_count
            scale_ind_range[s, :] = [start, end]

            patch_pts_valid += list(range(start, end))

            # convert points to torch tensors
            patch_pts[start:end, :] = torch.from_numpy(shape.pts[patch_point_inds, :])

            # center patch (central point at origin - but avoid changing padded zeros)
            if self.center == 'mean':
                patch_pts[start:end, :] = patch_pts[start:end, :] - patch_pts[start:end, :].mean(0)
            elif self.center == 'point':
                patch_pts[start:end, :] = patch_pts[start:end, :] - torch.from_numpy(shape.pts[center_point_ind, :])
            elif self.center == 'none':
                pass # no centering
            else:
                raise ValueError('Unknown patch centering option: %s' % (self.center))

            # normalize size of patch (scale with 1 / patch radius)
            patch_pts[start:end, :] = patch_pts[start:end, :] / rad


        if self.include_normals:
            patch_normal = torch.from_numpy(shape.normals[center_point_ind, :])

        if self.include_curvatures:
            patch_curv = torch.from_numpy(shape.curv[center_point_ind, :])
            # scale curvature to match the scaled vertices (curvature*s matches position/s):
            patch_curv = patch_curv * self.patch_radius_absolute[shape_ind][0]

        if self.use_pca:

            # compute pca of points in the patch:
            # center the patch around the mean:
            pts_mean = patch_pts[patch_pts_valid, :].mean(0)
            patch_pts[patch_pts_valid, :] = patch_pts[patch_pts_valid, :] - pts_mean

            trans, _, _ = torch.svd(torch.t(patch_pts[patch_pts_valid, :]))
            patch_pts[patch_pts_valid, :] = torch.mm(patch_pts[patch_pts_valid, :], trans)

            cp_new = -pts_mean # since the patch was originally centered, the original cp was at (0,0,0)
            cp_new = torch.matmul(cp_new, trans)

            # re-center on original center point
            patch_pts[patch_pts_valid, :] = patch_pts[patch_pts_valid, :] - cp_new

            if self.include_normals:
                patch_normal = torch.matmul(patch_normal, trans)

        else:
            trans = torch.eye(3).float()


        # get point tuples from the current patch
        if self.point_tuple > 1:
            patch_tuples = torch.zeros(self.points_per_patch*len(self.patch_radius_absolute[shape_ind]), 3*self.point_tuple, dtype=torch.float)
            for s, rad in enumerate(self.patch_radius_absolute[shape_ind]):
                start = scale_ind_range[s, 0]
                end = scale_ind_range[s, 1]
                point_count = end - start

                tuple_count = point_count**self.point_tuple

                # get linear indices of the tuples
                if tuple_count > self.points_per_patch:
                    patch_tuple_inds = self.rng.choice(tuple_count, self.points_per_patch, replace=False)
                    tuple_count = self.points_per_patch
                else:
                    patch_tuple_inds = np.arange(tuple_count)

                # linear tuple index to index for each tuple element
                patch_tuple_inds = np.unravel_index(patch_tuple_inds, (point_count,)*self.point_tuple)

                for t in range(self.point_tuple):
                    patch_tuples[start:start+tuple_count, t*3:(t+1)*3] = patch_pts[start+patch_tuple_inds[t], :]


            patch_pts = patch_tuples

        patch_feats = ()
        for pfeat in self.patch_features:
            if pfeat == 'normal':
                patch_feats = patch_feats + (patch_normal,)
            elif pfeat == 'max_curvature':
                patch_feats = patch_feats + (patch_curv[0:1],)
            elif pfeat == 'min_curvature':
                patch_feats = patch_feats + (patch_curv[1:2],)
            else:
                raise ValueError('Unknown patch feature: %s' % (pfeat))

        return (patch_pts,) + patch_feats + (trans,)


    def __len__(self):
        return sum(self.shape_patch_count)


