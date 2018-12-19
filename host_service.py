import click
from Preprocessing import fromITKtoNormalizedNumpy
from tomaat.server import TomaatApp, TomaatService
from tomaat.extras import TransformChain
from antsRegistration.ants import antsRegistration as ants
import time
input_interface = \
    [
        {'type': 'volume', 'destination': 'imageMoving'},
        {'type': 'volume', 'destination': 'imageFixed'}
    ]

output_interface = \
    [
        {'type': 'TransformGrid', 'field': 'transform'},
        {'type': 'TransformLinear', 'field': 'transformAffineMoving'},
        {'type': 'TransformLinear', 'field': 'transformAffineFixed'},
        {'type': 'PlainText', 'field': 'inference_time'}
    ]


def create_pre_process_pipeline(config):
    return TransformChain([])

def create_post_process_pipeline(config):
    return TransformChain([])


@click.group()
def cli():
    pass


@click.command()
def start_service():
    config = {
        "name": "VoxelMorph MICCAI 2018",
        "modality": "T1 MRI",
        "task": "Diffeomorphic Registration",
        "anatomy": "Brain",
        "description": "Example Description",
        "port": 8888,
        "announce": False,
        "api_key": "",

        "volume_resolution": [1, 1, 1],
        "volume_size": [160,200,160]
    }

    pre_process_pipeline = create_pre_process_pipeline(config)
    post_process_pipeline = create_post_process_pipeline(config)

    def prediction(data):
        import SimpleITK as sitk
        import tensorflow as tf
        import keras
        import numpy as np
        import os
        import shutil
        from DiffeomorphicRegistrationNet import create_model
        from DataGenerator import loadAtlas


        train_config = {
            'batchsize':1,
            'split':0.9,
            'validation':0.1,
            'half_res':True,
            'epochs': 200,
            'atlas': 'atlas.nii.gz',
            'model_output': 'model.pkl',
            'exponentialSteps': 7,
            'resolution':(160,200,160)
        }

        volFixed = sitk.ReadImage(data['imageFixed'][0])
        vol = sitk.ReadImage(data['imageMoving'][0])
        atlas_np,atlas = loadAtlas(train_config)

        # affine registration (preprocessing)
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(atlas)
        vol_atlas_resample = resampler.Execute(vol)
        volFixed_atlas_resample = resampler.Execute(volFixed)

        reg_dir = os.path.join(os.path.dirname(__file__),"registration_out")
        shutil.rmtree(reg_dir,ignore_errors=True)
        os.mkdir(reg_dir)
        registration_result = ants.registerImage(vol_atlas_resample,atlas,store_to=reg_dir,speed='fast')
        pre_trf = sitk.ReadTransform(registration_result['transforms_out'][0])

        reg_dirFixed = os.path.join(os.path.dirname(__file__),"registration_fixed_out")
        shutil.rmtree(reg_dirFixed,ignore_errors=True)
        os.mkdir(reg_dirFixed)
        registration_fixed_result = ants.registerImage(volFixed_atlas_resample,atlas,store_to=reg_dirFixed,speed='fast')
        preFixed_trf = sitk.ReadTransform(registration_fixed_result['transforms_out'][0])

        # resample with initial trf
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(atlas)
        resampler.SetTransform(pre_trf)
        registered_resampled = resampler.Execute(vol)

        resampler.SetTransform(preFixed_trf)
        registered_resampledFixed = resampler.Execute(volFixed)

        # get numpy arrays
        #fixed = fromITKtoNormalizedNumpy(atlas)
        fixed = fromITKtoNormalizedNumpy(registered_resampledFixed)
        moving = fromITKtoNormalizedNumpy(registered_resampled)

        vol_size = (160, 200, 160)

        fixed = fixed.reshape(vol_size)
        moving = moving.reshape(vol_size)

        from keras.backend import set_session
        set_session(tf.Session())

        # fixed parameters for network
        vol_size = (160, 200, 160)
        model = create_model(train_config)
        model.load_weights('weights-pairwise-v2.hdf5')

        inp = np.empty(160*200*160*3).reshape(1,160,200,160,3)
        inp[0,:,:,:,0] = atlas_np.reshape(vol_size)
        inp[0,:,:,:,1] = moving
        inp[0,:,:,:,2] = fixed

        #with tf.get_default_graph():
        p_start = time.time()
        pred = model.predict(inp)[0][0,:,:,:,:]
        p_end = time.time()

        warp_np = pred

        # reapply directions
        vol_dirs = np.array(atlas.GetDirection()).reshape(3,3)
        warp_np = np.flip(warp_np, [a for a in range(3) if vol_dirs[a, a] == -1.])
        # prepare axes swap from xyz to zyx
        print(warp_np.shape)
        warp_np = np.transpose(warp_np, (2, 1, 0, 3))
        # write image
        warp_img = sitk.GetImageFromArray(warp_np)
        warp_img.SetOrigin(atlas.GetOrigin())
        warp_img.SetDirection(atlas.GetDirection())
        #sitk.WriteImage(warp_img, "warptrf.nii.gz")

        result_data = {
            'transform':[warp_img],
            'transformAffineMoving': [pre_trf],
            'transformAffineFixed': [preFixed_trf],
            #'transformToAtlas': [trfToVMAtlasSpace],
            #'transformFromAtlas': [trfToCorrectSpace]
            'inference_time':["Inference time was: {}s".format(p_end-p_start)],
        }
        return result_data

    application = TomaatApp(
        preprocess_fun=pre_process_pipeline,
        inference_fun=prediction,
        postprocess_fun=post_process_pipeline
    )

    service = TomaatService(
        config=config,
        app=application,
        input_interface=input_interface,
        output_interface=output_interface
    )

    if config['announce']:
        service.start_service_announcement()

    service.run()


cli.add_command(start_service)


if __name__ == '__main__':
    cli()
