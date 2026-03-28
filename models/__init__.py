import inspect
import json
import warnings
from functools import lru_cache
from importlib import import_module
from pathlib import Path


warnings.filterwarnings(
    "ignore",
    message=r"Importing from timm\.models\.layers is deprecated, please import via timm\.layers",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"Importing from timm\.models\.registry is deprecated, please import via timm\.models",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"Overwriting pvt_v2_b[0-5] in registry with .*",
    category=UserWarning,
)


MODEL_REGISTRY = {
    "MEGANet": (".CNN.MEGANet_ResNet.EGANet", "eganet"),
    "SimpleUNet": (".CNN.SimpleUNet.SimpleUNet", "SimpleUNet"),
    "ULite": (".CNN.ULite.ULite", "ULite"),
    "MMUNet": (".CNN.MMUNet.MMUNet", "mmunet"),
    "UACANet": (".CNN.UACANet.UACANet", "UACANet"),
    "CSCAUNet": (".CNN.CSCAUNet.CSCAUNet", "CSCAUNet"),
    "UNetV2": (".CNN.UNet_v2.UNet_v2", "UNetV2"),
    "RollingUnet": (".CNN.RollingUnet.RollingUnet", "Rolling_Unet_M"),
    "DoubleUNet": (".CNN.dobuleunet.dobuleunet", "build_doubleunet"),
    "AttU_Net": (".CNN.AttU_Net.AttU_Net", "AttU_Net"),
    "CMUNeXt": (".CNN.CMUNeXt.CMUNeXt", "CMUNeXt"),
    "CMU_Net": (".CNN.CMU_Net.CMU_Net", "CMU_Net"),
    "UNeXt": (".CNN.UNeXt.UNeXt", "UNeXt"),
    "UNet3plus": (".CNN.UNet3plus.UNet3plus", "UNet3plus"),
    "ResNet34UnetPlus": (".CNN.UNetplus.UNetplus", "ResNet34UnetPlus"),
    "U_Net": (".CNN.U_Net.U_Net", "U_Net"),
    "Tinyunet": (".CNN.Tinyunet.Tinyunet", "Tinyunet"),
    "Egeunet": (".CNN.Egeunet.Egeunet", "EGEUNet"),
    "ERDUnet": (".CNN.ERDUnet.ERDUnet", "ERDUnet"),
    "MFMSNet": (".CNN.IS2D_models.mfmsnet", "MFMSNet"),
    "TA_Net": (".CNN.TA_Net.TA_Net", "TA_Net"),
    "DDANet": (".CNN.DDANet.DDANet", "ddanet"),
    "PraNet": (".CNN.PraNet.PraNet", "PraNet"),
    "ternausnet": (".CNN.TernausNet.TernausNet", "ternausnet"),
    "R2U_Net": (".CNN.R2U_Net.R2U_Net", "r2unet"),
    "CE_Net": (".CNN.CE_Net.CE_Net", "ce_net"),
    "MultiResUNet": (".CNN.MultiResUnet.MultiResUnet", "multiresunet"),
    "ResUNetPlusPlus": (".CNN.ResUnetPlusPlus.ResUnetPlusPlus", "resunetplusplus"),
    "MBSNet": (".CNN.MBSNet.MBSNet", "mbsnet"),
    "CA_Net": (".CNN.CA_Net.CA_Net", "ca_net"),
    "kiu_net": (".CNN.KiU_Net.KiU_Net", "kiu_net"),
    "LFU_Net": (".CNN.LFU_Net.LFU_Net", "lfu_net"),
    "DC_UNet": (".CNN.DC_Unet.DC_Unet", "dc_unet"),
    "ColonSegNet": (".CNN.ColonSegNet.ColonSegNet", "colonsegnet"),
    "MALUNet": (".CNN.MALUNet.MALUNet", "malunet"),
    "DCSAU_Net": (".CNN.DCSAU_Net.DCSAU_Net", "dcsau_net"),
    "FAT_Net": (".CNN.FAT_Net.FAT_Net", "fat_net"),
    "CFPNet_M": (".CNN.CFPNet_M.CFPNet_M", "cfpnet_m"),
    "CaraNet": (".CNN.CaraNet.CaraNet", "caranet"),
    "GH_UNet": (".CNN.GH_UNet.GH_UNet", "gh_unet"),
    "MSRFNet": (".CNN.MSRFNet.MSRFNet", "msrfnet"),
    "LV_UNet": (".CNN.LV_UNet.LV_UNet", "lv_unet"),
    "Perspective_Unet": (".CNN.Perspective_Unet.Perspective_Unet", "perspective_unet"),
    "ESKNet": (".CNN.ESKNet.ESKNet", "esknet"),
    "CPCANet": (".CNN.CPCANet.CPCANet", "cpcanet"),
    "UTANet": (".CNN.UTANet.UTANet", "utanet"),
    "DDS_UNet": (".CNN.DDS_UNet.DDS_UNet", "dds_unet"),
    "MCA_UNet": (".CNN.MCA_UNet.MCA_UNet", "mca_unet"),
    "MDSA_UNet": (".CNN.MDSA_UNet.MDSA_UNet", "mdsa_unet"),
    "U_KAN": (".CNN.U_KAN.U_KAN", "u_kan"),
    "ResU_KAN": (".CNN.ResU_KAN.ResU_KAN", "resu_kan"),
    "RAT_Net": (".CNN.RAT_Net.RAT_Net", "rat_net"),
    "AURA_Net": (".Hybrid.AURA_Net.AURA_Net", "aura_net"),
    "BEFUnet": (".Hybrid.BEFUnet.BEFUnet", "befunet"),
    "CASCADE": (".Hybrid.CASCADE.CASCADE", "cascade"),
    "G_CASCADE": (".Hybrid.G_CASCADE.G_CASCADE", "g_cascade"),
    "ConvFormer": (".Hybrid.ConvFormer.ConvFormer", "convformer"),
    "DA_TransUNet": (".Hybrid.DA_TransUNet.DA_TransUNet", "da_transformer"),
    "DAEFormer": (".Hybrid.DAEFormer.DAEFormer", "daeformer"),
    "DS_TransUNet": (".Hybrid.DS_TransUNet.DS_TransUNet", "ds_transunet"),
    "FCBFormer": (".Hybrid.FCBFormer.FCBFormer", "fcbformer"),
    "HiFormer": (".Hybrid.HiFormer.HiFormer", "hiformer"),
    "LeViT_UNet": (".Hybrid.LeViT_UNet.LeViT_UNet", "levit_unet"),
    "MERIT": (".Hybrid.MERIT.MERIT", "merit"),
    "MT_UNet": (".Hybrid.MT_UNet.MT_UNet", "mt_unet"),
    "TransAttUnet": (".Hybrid.TransAttUnet.TransAttUnet", "trans_attention_unet"),
    "TransFuse": (".Hybrid.TransFuse.TransFuse", "transfuse"),
    "TransNorm": (".Hybrid.TransNorm.TransNorm", "transnorm"),
    "TransResUNet": (".Hybrid.TransResUNet.TransResUNet", "trans_res_unet"),
    "UTNet": (".Hybrid.UTNet.UTNet", "utnet"),
    "UCTransNet": (".Hybrid.UCTransNet.UCTransNet", "UCTransNet"),
    "EMCAD": (".Hybrid.EMCAD.networks", "EMCADNet"),
    "CSWin_UNet": (".Hybrid.CSWin_UNet.CSWin_UNet", "cswin_unet"),
    "D_TrAttUnet": (".Hybrid.D_TrAttUnet.D_TrAttUnet", "d_trattunet"),
    "EViT_UNet": (".Hybrid.EViT_UNet.EViT_UNet", "evit_unet"),
    "MedFormer": (".Hybrid.MedFormer.MedFormer", "medformer"),
    "MSLAU_Net": (".Hybrid.MSLAU_Net.MSLAU_Net", "mslau_net"),
    "MissFormer": (".Hybrid.MissFormer.MissFormer", "Missformer"),
    "TransUnet": (".Hybrid.TransUnet.TransUnet", "transunet"),
    "MobileUViT": (".Hybrid.MobileUViT.MobileUViT", "mobileuvit_l"),
    "LGMSNet": (".Hybrid.LGMSNet.LGMSNet", "lgmsnet"),
    "SwinUNETR": (".Hybrid.SwinUNETR.SwinUNETR", "swinunetr"),
    "UNETR": (".Hybrid.UNETR.UNETR", "unetr"),
    "CFFormer": (".Hybrid.CFFormer.CFFormer", "cfformer"),
    "CENet": (".Hybrid.CENet.CENet", "cenet"),
    "H2Former": (".Hybrid.H2Former.H2Former", "h2former"),
    "ScribFormer": (".Hybrid.ScribFormer.ScribFormer", "scribformer"),
    "BATFormer": (".Transformer.BATFormer.BATFormer", "batformer"),
    "Polyp_PVT": (".Transformer.Polyp_PVT.Polyp_PVT", "polyp_pvt"),
    "SCUNet_plus_plus": (".Transformer.SCUNet_plus_plus.SCUNet_plus_plus", "scunet_plus_plus"),
    "SwinUnet": (".Transformer.SwinUnet.SwinUnet", "swinunet"),
    "MedT": (".Transformer.MedT.MedT", "medt"),
    "AC_MambaSeg": (".Mamba.AC_MambaSeg.AC_MambaSeg", "ac_mambaseg"),
    "H_vmunet": (".Mamba.H_vmunet.H_vmunet", "h_vmunet"),
    "MambaUnet": (".Mamba.MambaUnet.MambaUnet", "mambaunet"),
    "MUCM_Net": (".Mamba.MUCM_Net.MUCM_Net", "mucm_net"),
    "Swin_umamba": (".Mamba.Swin_umamba.Swin_umamba", "swin_umamba"),
    "Swin_umambaD": (".Mamba.Swin_umambaD.Swin_umambaD", "swin_umambad"),
    "UltraLight_VM_UNet": (".Mamba.UltraLight_VM_UNet.UltraLight_VM_UNet", "ultralight_vm_unet"),
    "VMUNet": (".Mamba.VMUNet.VMUNet", "vmunet"),
    "VMUNetV2": (".Mamba.VMUNetV2.VMUNetV2", "vmunetv2"),
    "CFM_UNet": (".Mamba.CFM_UNet.CFM_UNet", "cfm_unet"),
    "MedVKAN": (".Mamba.MedVKAN.MedVKAN", "medvkan"),
    "Zig_RiR": (".RWKV.Zig_RiR.Zig_RiR", "zig_rir"),
    "RWKV_UNet": (".RWKV.RWKV_UNet.RWKV_UNet", "rwkv_unet"),
    "U_RWKV": (".RWKV.U_RWKV.U_RWKV", "u_rwkv"),
}


MODEL_ID_PATH = Path(__file__).with_name("model_id.json")


def available_models():
    return sorted(MODEL_REGISTRY)


@lru_cache(maxsize=1)
def _load_model_metadata():
    with MODEL_ID_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


@lru_cache(maxsize=None)
def _load_model_factory(model_name):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Model '{model_name}' is not registered. Available models: {available_models()}"
        )

    module_path, attr_name = MODEL_REGISTRY[model_name]
    try:
        module = import_module(module_path, package=__name__)
    except ModuleNotFoundError as exc:
        missing_module = exc.name or "unknown"
        raise ModuleNotFoundError(
            f"Model '{model_name}' could not be imported from '{module_path}'. "
            f"Missing module: '{missing_module}'. Install that dependency only if you need this model."
        ) from exc

    try:
        return getattr(module, attr_name)
    except AttributeError as exc:
        raise AttributeError(
            f"Model '{model_name}' is registered to '{module_path}:{attr_name}', "
            "but that symbol was not found."
        ) from exc


def load_model_id(model_name):
    for model_info in _load_model_metadata():
        if model_info["modelname"] == model_name:
            deep_supervision = model_info.get("deeps_supervision", 0)
            model_id = model_info.get("id")
            if model_id is None:
                raise ValueError(f"Model '{model_name}' does not have a valid model_id.")
            return model_id, deep_supervision

    raise ValueError(f"Model '{model_name}' not found in {MODEL_ID_PATH}.")


def _instantiate_model(model_factory, kwargs):
    try:
        signature = inspect.signature(model_factory)
    except (TypeError, ValueError):
        return model_factory(**kwargs)

    accepts_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_var_kwargs:
        return model_factory(**kwargs)

    unexpected_kwargs = [
        key for key in kwargs if key not in signature.parameters
    ]
    if unexpected_kwargs:
        factory_name = getattr(model_factory, "__name__", repr(model_factory))
        raise TypeError(
            f"Model factory '{factory_name}' does not accept keyword arguments: {unexpected_kwargs}"
        )

    return model_factory(**kwargs)


def build_model(config, **kwargs):
    model_name = config.model
    model_id, config.do_deeps = load_model_id(model_name)
    print(f"Building model {model_name} with model_id {model_id} and do_deeps {config.do_deeps}")
    config.model_id = model_id
    print(f"Using model_id {model_id} for model {model_name}")

    if (
        model_name == "RWKV_UNet"
        and hasattr(config, "img_size")
        and "img_size" not in kwargs
    ):
        kwargs = dict(kwargs)
        kwargs["img_size"] = int(config.img_size)

    model_factory = _load_model_factory(model_name)
    print(f"kwargs: {kwargs}")
    return _instantiate_model(model_factory, kwargs)


__all__ = ["available_models", "build_model", "load_model_id"]
