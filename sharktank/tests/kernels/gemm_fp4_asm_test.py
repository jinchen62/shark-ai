# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import torch
import unittest
import iree.turbine.aot as aot
from sharktank.kernels.gemm_fp4_asm import asm_fp4_gemm

logging.basicConfig(level=logging.DEBUG)


class TestAsmFp4Gemm(unittest.TestCase):
    def test_asm_fp4_gemm(self):
        class AsmMxfp4GemmModule(torch.nn.Module):
            def forward(self, x, w, x_scale, w_scale, bias):
                return asm_fp4_gemm(x, w, x_scale, w_scale, bias)

        e = aot.export(
            AsmMxfp4GemmModule(),
            args=(
                torch.empty((256, 512), dtype=torch.uint8),
                torch.empty((256, 512), dtype=torch.uint8),
                torch.empty((256, 32), dtype=torch.uint8),
                torch.empty((256, 32), dtype=torch.uint8),
                torch.empty((256, 256), dtype=torch.float32),
            ),
        )
        e.verify()
        mlir_asm = str(e.mlir_module)
        print(mlir_asm)
        # self.assertIn(
        #     ("func.func @main"),
        #     mlir_asm,
        # )
        # self.assertIn(
        #     ("stream.executable private @batched_gemm"),
        #     mlir_asm,
        # )
        # self.assertIn(
        #     (
        #         "func.func private @wave_mxfp4_bmm_B_dyn_M_dyn_HALF_K_512_u8_B_dyn_M_dyn_K_OVER_THIRTYTWO_32_u8_N_1024_HALF_K512_u8_N_1024_K_OVER_THIRTYTWO_32_u8_B_dyn_M_dyn_N_1024_f32"
        #     ),
        #     mlir_asm,
        # )
        # self.assertIn(
        #     (
        #         "util.func private @wave_mxfp4_bmm_B_M_HALF_K_512_i8_B_M_K_OVER_THIRTYTWO_32_i8_N_1024_HALF_K_512_i8_N_1024_K_OVER_THIRTYTWO_32_i8_B_M_N_1024_f32_B_M_N_1024_f32"
        #     ),
        #     mlir_asm,
        # )


if __name__ == "__main__":
    unittest.main()
