from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/home/yan/tracking/Stark/data/got10k_lmdb'
    settings.got10k_path = '/home/yan/tracking/Stark/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_lmdb_path = '/home/yan/tracking/Stark/data/lasot_lmdb'
    settings.lasot_path = '/home/yan/tracking/Stark/data/lasot'
    settings.network_path = '/home/yan/tracking/Stark/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/home/yan/tracking/Stark/data/nfs'
    settings.otb_path = '/home/yan/data/OTB100'
    settings.prj_dir = '/home/yan/tracking/Stark'
    settings.result_plot_path = '/home/yan/tracking/Stark/test/result_plots'
    settings.results_path = '/home/yan/tracking/Stark/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/home/yan/tracking/Stark'
    settings.segmentation_path = '/home/yan/tracking/Stark/test/segmentation_results'
    settings.tc128_path = '/home/yan/tracking/Stark/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = '/home/yan/tracking/Stark/data/trackingNet'
    settings.uav_path = '/home/yan/tracking/Stark/data/UAV123'
    settings.vot_path = '/home/yan/tracking/Stark/data/VOT2019'
    settings.youtubevos_dir = ''

    return settings

