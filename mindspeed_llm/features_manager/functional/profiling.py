# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
from argparse import ArgumentParser

from mindspeed.features_manager.feature import MindSpeedFeature


class ProfilingFeature(MindSpeedFeature):

    def __init__(self):
        super(ProfilingFeature, self).__init__(feature_name="profiling", optimization_level=0)

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument('--profile-export-type', type=str, default='text',
                                         choices=['text', 'db'], help='choose the export mode as text or db.')
        group.add_argument('--profile-level', type=str, default='level0',
                                         choices=['level_none', 'level0', 'level1', 'level2'],
                                         help='profiling level_none, level0, level1, level2')
        group.add_argument('--profile-data-simplification', action='store_true',
                                         help='use data simplification mode')
        group.add_argument('--profile-with-stack', action='store_true',
                                         help='profiling with stack info')
        group.add_argument('--profile-with-memory', action='store_true',
                                         help='profiling with memory info')
        group.add_argument('--profile-record-shapes', action='store_true',
                                         help='profiling with shape info')
        group.add_argument('--profile-with-cpu', action='store_true',
                                         help='profiling with cpu info')
        group.add_argument('--profile-save-path', type=str, default='./profile_dir',
                                         help='path to save profiling files')
        # --- Chakra 通算 DAG 导出 ---
        # 采集结构（运行时钩子）+ 补时长（profiling trace）→ Chakra execution trace。
        group.add_argument('--chakra-dag-export', action='store_true',
                                         help='export a compute/communication DAG in Chakra format after profiling')
        group.add_argument('--chakra-dag-output-path', type=str, default=None,
                                         help='output directory for DAG artifacts; default <profile-save-path>/chakra_dag')
        group.add_argument('--chakra-dag-format', type=str, default='both',
                                         choices=['protobuf', 'json', 'both'],
                                         help="DAG output format; 'json' does not require the chakra package")
        group.add_argument('--chakra-dag-fused-mode', type=str, default='split-estimate',
                                         choices=['split-measured', 'split-estimate', 'merged'],
                                         help='how to split the measured duration of a fused MC2/CoC operator')
        group.add_argument('--chakra-dag-bandwidth-gbps', type=float, default=None,
                                         help='bandwidth in GB/s used to estimate the comm part of a fused operator')
        group.add_argument('--chakra-dag-inplace', action='store_true',
                                         help='run DAG export in the current process instead of a subprocess (debug)')
        group.add_argument('--chakra-dag-timeout-seconds', type=float, default=1800.0,
                                         help='timeout for the DAG export subprocess')
        group.add_argument('--chakra-dag-max-events', type=int, default=2000000,
                                         help='upper bound on captured events per rank')

    def validate_args(self, args):
        super().validate_args(args)

        if hasattr(args, 'profile_step_start') and args.profile_step_start < 1:
            raise AssertionError(f'profile_step_start should be greater than 0 but now it is {args.profile_step_start}')
