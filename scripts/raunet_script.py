import logging
import sys
import traceback
from typing import Any
from functools import partial
import gradio as gr
from modules import script_callbacks, scripts
from HiDiffusion.raunet import ApplyRAUNet, ApplyRAUNetSimple, UPSCALE_METHODS
from HiDiffusion.msw_msa_attention import ApplyMSWMSAAttention, ApplyMSWMSAAttentionSimple

logging.debug("Imports successful in RAUNet script")
opApplyRAUNet = ApplyRAUNet()
opApplyRAUNetSimple = ApplyRAUNetSimple()
opApplyMSWMSA = ApplyMSWMSAAttention()
opApplyMSWMSASimple = ApplyMSWMSAAttentionSimple()
class RAUNetScript(scripts.Script):
    def __init__(self):
        self.raunet_was_enabled = False  # Track RAUNet state
        self.mswmsa_was_enabled = False  # Track MSW-MSA state

    sorting_priority = 16.05

    def title(self):
        return "RAUNet/MSW-MSA for reForge"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with gr.Accordion(open=False, label=self.title()):
            gr.HTML("<p><i>Make sure to use only either the simple or the advanced version.</i></p>")
            with gr.Tab("RAUNet Simple"):
                gr.Markdown("Simplified RAUNet for easier setup. Helps avoid artifacts at high resolutions.")
                raunet_simple_enabled = gr.Checkbox(label="RAUNet Simple Enabled", value=False)
                raunet_simple_model_type = gr.Radio(choices=["SD15", "SDXL"], value="SD15", label="Model Type")
                gr.Markdown("Note: Use SD15 setting for SD 2.1 as well.")
                res_mode = gr.Radio(choices=["high (1536-2048)", "low (1024 or lower)", "ultra (over 2048)"], value="high (1536-2048)", label="Resolution Mode")
                gr.Markdown("Note: Resolution mode is a preset, exact match to your resolution is not necessary.")
                simple_upscale_mode = gr.Dropdown(choices=["default"] + list(UPSCALE_METHODS), value="default", label="Upscale Mode")
                simple_ca_upscale_mode = gr.Dropdown(choices=["default"] + list(UPSCALE_METHODS), value="default", label="CA Upscale Mode")

            with gr.Tab("RAUNet Advanced"):
                gr.Markdown("Advanced RAUNet settings. For fine-tuning artifact reduction at high resolutions.")
                raunet_enabled = gr.Checkbox(label="RAUNet Enabled", value=False)
                raunet_model_type = gr.Radio(choices=["SD15", "SDXL"], value="SD15", label="Model Type")
                gr.Markdown("Note: Use SD15 setting for SD 2.1 as well.")
                input_blocks = gr.Text(label="Input Blocks", value="3")
                output_blocks = gr.Text(label="Output Blocks", value="8")
                gr.Markdown("For SD1.5: Input 3 corresponds to Output 8, Input 6 to Output 5, Input 9 to Output 2")
                gr.Markdown("For SDXL: Input 3 corresponds to Output 5, Input 6 to Output 2")
                time_mode = gr.Dropdown(choices=["percent", "timestep", "sigma"], value="percent", label="Time Mode")
                gr.Markdown("Time mode: Controls format of start/end times. Use percent if unsure.")
                start_time = gr.Slider(label="Start Time", minimum=0.0, maximum=1.0, step=0.01, value=0.0)
                end_time = gr.Slider(label="End Time", minimum=0.0, maximum=1.0, step=0.01, value=0.45)
                skip_two_stage_upscale = gr.Checkbox(label="Skip Two Stage Upscale", value=False)
                upscale_mode = gr.Dropdown(choices=UPSCALE_METHODS, value="bicubic", label="Upscale Mode")
                gr.Markdown("Recommended upscale mode: bicubic or bislerp")
                
                with gr.Accordion(open=False, label="Cross-Attention Settings"):
                    ca_start_time = gr.Slider(label="CA Start Time", minimum=0.0, maximum=1.0, step=0.01, value=0.0)
                    ca_end_time = gr.Slider(label="CA End Time", minimum=0.0, maximum=1.0, step=0.01, value=0.3)
                    ca_input_blocks = gr.Text(label="CA Input Blocks", value="4")
                    ca_output_blocks = gr.Text(label="CA Output Blocks", value="8")
                    ca_upscale_mode = gr.Dropdown(choices=UPSCALE_METHODS, value="bicubic", label="CA Upscale Mode")

            with gr.Tab("MSW-MSA Simple"):
                gr.Markdown("Simplified MSW-MSA for easier setup. Can improve performance and quality at high resolutions.")
                mswmsa_simple_enabled = gr.Checkbox(label="MSW-MSA Simple Enabled", value=False)
                mswmsa_simple_model_type = gr.Radio(choices=["SD15", "SDXL"], value="SD15", label="Model Type")
                gr.Markdown("Note: Use SD15 setting for SD 2.1 as well.")

            with gr.Tab("MSW-MSA Advanced"):
                gr.Markdown("Advanced MSW-MSA settings. For fine-tuning performance and quality improvements.")
                mswmsa_enabled = gr.Checkbox(label="MSW-MSA Enabled", value=False)
                mswmsa_model_type = gr.Radio(choices=["SD15", "SDXL"], value="SD15", label="Model Type")
                gr.Markdown("Note: Use SD15 setting for SD 2.1 as well.")
                mswmsa_input_blocks = gr.Text(label="Input Blocks", value="1,2")
                mswmsa_middle_blocks = gr.Text(label="Middle Blocks", value="")
                mswmsa_output_blocks = gr.Text(label="Output Blocks", value="9,10,11")
                gr.Markdown("Recommended SD15: input 1,2, output 9,10,11")
                gr.Markdown("Recommended SDXL: input 4,5, output 4,5")
                mswmsa_time_mode = gr.Dropdown(choices=["percent", "timestep", "sigma"], value="percent", label="Time Mode")
                mswmsa_start_time = gr.Slider(label="Start Time", minimum=0.0, maximum=1.0, step=0.01, value=0.0)
                mswmsa_end_time = gr.Slider(label="End Time", minimum=0.0, maximum=1.0, step=0.01, value=1.0)
                gr.Markdown("Note: For very high resolutions (>2048), try starting at 0.2 or after other scaling effects end.")

            gr.HTML("<p><i>Note: Make sure you use the options corresponding to your model type (SD1.5 or SDXL). Otherwise, it may have no effect or fail.</i></p>")
            gr.Markdown("Compatibility: These methods may not work with other attention modifications or scaling effects targeting the same blocks.")

        # Add JavaScript to handle visibility and model-specific settings
        def update_raunet_settings(model_type):
            if model_type == "SD15":
                return "3", "8", "4", "8", 0.0, 0.45, 0.0, 0.3
            else:  # SDXL
                return "3", "5", "2", "7", 1.0, 1.0, 1.0, 1.0  # Disabling both patches by default for SDXL

        raunet_model_type.change(
            fn=update_raunet_settings,
            inputs=[raunet_model_type],
            outputs=[input_blocks, output_blocks, ca_input_blocks, ca_output_blocks, start_time, end_time, ca_start_time, ca_end_time]
        )

        def update_mswmsa_settings(model_type):
            if model_type == "SD15":
                return "1,2", "", "9,10,11"
            else:  # SDXL
                return "4,5", "", "4,5"

        mswmsa_model_type.change(
            fn=update_mswmsa_settings,
            inputs=[mswmsa_model_type],
            outputs=[mswmsa_input_blocks, mswmsa_middle_blocks, mswmsa_output_blocks]
        )

        return (raunet_simple_enabled, raunet_simple_model_type, res_mode, simple_upscale_mode, simple_ca_upscale_mode,
            raunet_enabled, raunet_model_type, input_blocks, output_blocks, time_mode, start_time, end_time, 
            skip_two_stage_upscale, upscale_mode, ca_start_time, ca_end_time, ca_input_blocks, ca_output_blocks, ca_upscale_mode,
            mswmsa_simple_enabled, mswmsa_simple_model_type,
            mswmsa_enabled, mswmsa_model_type, mswmsa_input_blocks, mswmsa_middle_blocks, mswmsa_output_blocks, 
            mswmsa_time_mode, mswmsa_start_time, mswmsa_end_time)

    def process_before_every_sampling(self, p, *script_args, **kwargs):
        (raunet_simple_enabled, raunet_simple_model_type, res_mode, simple_upscale_mode, simple_ca_upscale_mode,
        raunet_enabled, raunet_model_type, input_blocks, output_blocks, time_mode, start_time, end_time, 
        skip_two_stage_upscale, upscale_mode, ca_start_time, ca_end_time, ca_input_blocks, ca_output_blocks, ca_upscale_mode,
        mswmsa_simple_enabled, mswmsa_simple_model_type,
        mswmsa_enabled, mswmsa_model_type, mswmsa_input_blocks, mswmsa_middle_blocks, mswmsa_output_blocks, 
        mswmsa_time_mode, mswmsa_start_time, mswmsa_end_time) = script_args

        # Retrieve values from XYZ plot if available
        xyz = getattr(p, "_raunet_xyz", {})
        
        # Handle RAUNet Simple XYZ values
        if "raunet_simple_enabled" in xyz:
            raunet_simple_enabled = xyz["raunet_simple_enabled"] == "True"
        if "raunet_simple_model_type" in xyz:
            raunet_simple_model_type = xyz["raunet_simple_model_type"]
        if "res_mode" in xyz:
            res_mode = xyz["res_mode"]
        if "simple_upscale_mode" in xyz:
            simple_upscale_mode = xyz["simple_upscale_mode"]
        if "simple_ca_upscale_mode" in xyz:
            simple_ca_upscale_mode = xyz["simple_ca_upscale_mode"]

        # Handle RAUNet Advanced XYZ values
        if "raunet_enabled" in xyz:
            raunet_enabled = xyz["raunet_enabled"] == "True"
        if "raunet_model_type" in xyz:
            raunet_model_type = xyz["raunet_model_type"]
        if "start_time" in xyz:
            start_time = xyz["start_time"]
        if "end_time" in xyz:
            end_time = xyz["end_time"]

        # Handle MSW-MSA Simple XYZ values
        if "mswmsa_simple_enabled" in xyz:
            mswmsa_simple_enabled = xyz["mswmsa_simple_enabled"] == "True"
        if "mswmsa_simple_model_type" in xyz:
            mswmsa_simple_model_type = xyz["mswmsa_simple_model_type"]

        # Handle MSW-MSA Advanced XYZ values
        if "mswmsa_enabled" in xyz:
            mswmsa_enabled = xyz["mswmsa_enabled"] == "True"
        if "mswmsa_model_type" in xyz:
            mswmsa_model_type = xyz["mswmsa_model_type"]
        if "mswmsa_start_time" in xyz:
            mswmsa_start_time = xyz["mswmsa_start_time"]
        if "mswmsa_end_time" in xyz:
            mswmsa_end_time = xyz["mswmsa_end_time"]

        # Always start with a fresh clone of the original unet
        unet = p.sd_model.forge_objects.unet.clone()

        # Handle RAUNet reset if needed
        raunet_is_enabled = raunet_simple_enabled or raunet_enabled
        if not raunet_is_enabled and self.raunet_was_enabled:
            # Reset RAUNet modifications
            unet = opApplyRAUNet.patch(False, unet, "", "", "", 0, 0, False, "", 0, 0, "", "", "")[0]
            unet = opApplyRAUNetSimple.go(False, raunet_simple_model_type, res_mode, simple_upscale_mode, simple_ca_upscale_mode, unet)[0]
            self.raunet_was_enabled = False

        # Handle MSW-MSA reset if needed
        # For now it is bugged on first gen after disabling it, but consequent gens will work.
        # TODO: Fix this
        mswmsa_is_enabled = mswmsa_simple_enabled or mswmsa_enabled
        if not mswmsa_is_enabled and self.mswmsa_was_enabled:
            # Reset MSW-MSA modifications
            unet = opApplyMSWMSA.patch(unet, "", "", "", mswmsa_time_mode, 0, 0)[0]
            unet = opApplyMSWMSASimple.go(mswmsa_simple_model_type, unet)[0]
            self.mswmsa_was_enabled = False

        # Handle RAUNet if enabled
        if raunet_simple_enabled:
            unet = opApplyRAUNetSimple.go(
                True, raunet_simple_model_type, res_mode, simple_upscale_mode, simple_ca_upscale_mode, unet
            )[0]
            self.raunet_was_enabled = True
            p.extra_generation_params.update(
                dict(
                    raunet_simple_enabled=True,
                    raunet_model_type=raunet_simple_model_type,
                    raunet_res_mode=res_mode,
                    raunet_simple_upscale_mode=simple_upscale_mode,
                    raunet_simple_ca_upscale_mode=simple_ca_upscale_mode,
                )
            )
        elif raunet_enabled:
            unet = opApplyRAUNet.patch(
                True, unet, input_blocks, output_blocks, time_mode, start_time, end_time, skip_two_stage_upscale, upscale_mode,
                ca_start_time, ca_end_time, ca_input_blocks, ca_output_blocks, ca_upscale_mode
            )[0]
            self.raunet_was_enabled = True
            p.extra_generation_params.update(
                dict(
                    raunet_enabled=True,
                    raunet_model_type=raunet_model_type,
                    raunet_input_blocks=input_blocks,
                    raunet_output_blocks=output_blocks,
                    raunet_time_mode=time_mode,
                    raunet_start_time=start_time,
                    raunet_end_time=end_time,
                    raunet_skip_two_stage_upscale=skip_two_stage_upscale,
                    raunet_upscale_mode=upscale_mode,
                    raunet_ca_start_time=ca_start_time,
                    raunet_ca_end_time=ca_end_time,
                    raunet_ca_input_blocks=ca_input_blocks,
                    raunet_ca_output_blocks=ca_output_blocks,
                    raunet_ca_upscale_mode=ca_upscale_mode,
                )
            )
        else:
            # Apply RAUNet patch with enabled=False to reset any modifications
            unet = opApplyRAUNet.patch(False, unet, "", "", "", 0, 0, False, "", 0, 0, "", "", "")[0]
            unet = opApplyRAUNetSimple.go(False, raunet_simple_model_type, res_mode, simple_upscale_mode, simple_ca_upscale_mode, unet)[0]
            p.extra_generation_params.update(dict(raunet_enabled=False, raunet_simple_enabled=False))

        # Handle MSW-MSA if enabled
        if mswmsa_simple_enabled:
            unet = opApplyMSWMSASimple.go(mswmsa_simple_model_type, unet)[0]
            self.mswmsa_was_enabled = True
            p.extra_generation_params.update(
                dict(
                    mswmsa_simple_enabled=True,
                    mswmsa_model_type=mswmsa_simple_model_type,
                )
            )
        elif mswmsa_enabled:
            unet = opApplyMSWMSA.patch(
                unet, mswmsa_input_blocks, mswmsa_middle_blocks, mswmsa_output_blocks, mswmsa_time_mode, mswmsa_start_time, mswmsa_end_time
            )[0]
            self.mswmsa_was_enabled = True
            p.extra_generation_params.update(
                dict(
                    mswmsa_enabled=True,
                    mswmsa_model_type=mswmsa_model_type,
                    mswmsa_input_blocks=mswmsa_input_blocks,
                    mswmsa_middle_blocks=mswmsa_middle_blocks,
                    mswmsa_output_blocks=mswmsa_output_blocks,
                    mswmsa_time_mode=mswmsa_time_mode,
                    mswmsa_start_time=mswmsa_start_time,
                    mswmsa_end_time=mswmsa_end_time,
                )
            )
        else:
            # Apply MSW-MSA patch with empty block settings to reset any modifications
            unet = opApplyMSWMSA.patch(unet, "", "", "", mswmsa_time_mode, 0, 0)[0]
            unet = opApplyMSWMSASimple.go(mswmsa_simple_model_type, unet)[0]
            p.extra_generation_params.update(dict(mswmsa_enabled=False, mswmsa_simple_enabled=False))

        # Always update the unet
        p.sd_model.forge_objects.unet = unet
        # Add debug logging
        logging.debug(f"RAUNet Simple enabled: {raunet_simple_enabled}, Model Type: {raunet_simple_model_type}")
        logging.debug(f"RAUNet enabled: {raunet_enabled}, Model Type: {raunet_model_type}")
        logging.debug(f"MSW-MSA Simple enabled: {mswmsa_simple_enabled}, Model Type: {mswmsa_simple_model_type}")
        logging.debug(f"MSW-MSA enabled: {mswmsa_enabled}, Model Type: {mswmsa_model_type}")

        return
    
def set_value(p, x: Any, xs: Any, *, field: str):
    if not hasattr(p, "_raunet_xyz"):
        p._raunet_xyz = {}
    p._raunet_xyz[field] = x

def make_axis_on_xyz_grid():
    xyz_grid = None
    for script in scripts.scripts_data:
        if script.script_class.__module__ == "xyz_grid.py":
            xyz_grid = script.module
            break
    
    if xyz_grid is None:
        return

    axis = [
        # RAUNet Simple options
        xyz_grid.AxisOption(
            "(RAUNet) Simple Enabled",
            str,
            partial(set_value, field="raunet_simple_enabled"),
            choices=lambda: ["True", "False"]
        ),
        xyz_grid.AxisOption(
            "(RAUNet) Simple Model Type",
            str,
            partial(set_value, field="raunet_simple_model_type"),
            choices=lambda: ["SD15", "SDXL"]
        ),
        xyz_grid.AxisOption(
            "(RAUNet) Resolution Mode",
            str,
            partial(set_value, field="res_mode"),
            choices=lambda: ["high (1536-2048)", "low (1024 or lower)", "ultra (over 2048)"]
        ),
        xyz_grid.AxisOption(
            "(RAUNet) Simple Upscale Mode",
            str,
            partial(set_value, field="simple_upscale_mode"),
            choices=lambda: ["default"] + list(UPSCALE_METHODS)
        ),
        xyz_grid.AxisOption(
            "(RAUNet) Simple CA Upscale Mode",
            str,
            partial(set_value, field="simple_ca_upscale_mode"),
            choices=lambda: ["default"] + list(UPSCALE_METHODS)
        ),

        # RAUNet Advanced options
        xyz_grid.AxisOption(
            "(RAUNet) Advanced Enabled",
            str,
            partial(set_value, field="raunet_enabled"),
            choices=lambda: ["True", "False"]
        ),
        xyz_grid.AxisOption(
            "(RAUNet) Advanced Model Type",
            str,
            partial(set_value, field="raunet_model_type"),
            choices=lambda: ["SD15", "SDXL"]
        ),
        xyz_grid.AxisOption(
            "(RAUNet) Start Time",
            float,
            partial(set_value, field="start_time"),
        ),
        xyz_grid.AxisOption(
            "(RAUNet) End Time",
            float,
            partial(set_value, field="end_time"),
        ),

        # MSW-MSA Simple options
        xyz_grid.AxisOption(
            "(MSW-MSA) Simple Enabled",
            str,
            partial(set_value, field="mswmsa_simple_enabled"),
            choices=lambda: ["True", "False"]
        ),
        xyz_grid.AxisOption(
            "(MSW-MSA) Simple Model Type",
            str,
            partial(set_value, field="mswmsa_simple_model_type"),
            choices=lambda: ["SD15", "SDXL"]
        ),

        # MSW-MSA Advanced options
        xyz_grid.AxisOption(
            "(MSW-MSA) Advanced Enabled",
            str,
            partial(set_value, field="mswmsa_enabled"),
            choices=lambda: ["True", "False"]
        ),
        xyz_grid.AxisOption(
            "(MSW-MSA) Advanced Model Type",
            str,
            partial(set_value, field="mswmsa_model_type"),
            choices=lambda: ["SD15", "SDXL"]
        ),
        xyz_grid.AxisOption(
            "(MSW-MSA) Start Time",
            float,
            partial(set_value, field="mswmsa_start_time"),
        ),
        xyz_grid.AxisOption(
            "(MSW-MSA) End Time",
            float,
            partial(set_value, field="mswmsa_end_time"),
        ),
    ]

    if not any(x.label.startswith("(RAUNet)") or x.label.startswith("(MSW-MSA)") for x in xyz_grid.axis_options):
        xyz_grid.axis_options.extend(axis)

def on_before_ui():
    try:
        make_axis_on_xyz_grid()
    except Exception:
        error = traceback.format_exc()
        print(
            f"[-] RAUNet/MSW-MSA Script: xyz_grid error:\n{error}",
            file=sys.stderr,
        )

script_callbacks.on_before_ui(on_before_ui)