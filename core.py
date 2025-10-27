import numpy as np
import colour # type: ignore

def build_rosch_macadam_XYZ(
    cmfs_name='CIE 1931 2 Degree Standard Observer',
    illuminant_name='D65',
    bins=None,
    point_order='Pulse Wave Width',
    filter_jagged_points=False,
    spectral_res=None # default is 5 nm. Otherwise only 1, 5, 10 or 20 nm allowed
):
    """
    Returns:
    XYZ_rel   : (N, 3) XYZ with Y_white = 1 (relative luminance factor)
    Lab_ref   : (N, 3) CIELAB (L* 0..100, a*, b* in conventional units)
    rgb_phys  : (N, 3) sRGB (gamma) from XYZ_rel (physically scaled; often dark)
    rgb_bright: (N, 3) sRGB (gamma) from exposure-scaled XYZ_rel (for display)
    hue_deg   : (N,)   CIELAB hue angle in degrees
    chroma    : (N,)   CIELAB chroma C*ab
    debug     : dict   helpful ranges and configuration
    """
    colour.set_domain_range_scale('1')
    try:
        colour.utilities.CACHE_REGISTRY.clear()
    except Exception:
        pass

    if spectral_res is None:
        shape = colour.volume.spectrum.SPECTRAL_SHAPE_OUTER_SURFACE_XYZ
    else:
        try:
            spec_val = float(spectral_res)
            allowed = np.array([1, 5, 10, 20], dtype=float)
            spectral_res = int(allowed[np.abs(allowed - spec_val).argmin()])
        except Exception:
            spectral_res = 5
        shape = colour.SpectralShape(360, 780, spectral_res)
    cmfs = colour.MSDS_CMFS[cmfs_name].copy().align(shape)
    illuminant = colour.SDS_ILLUMINANTS[illuminant_name].copy().align(shape)


    if bins is None:
        bins = cmfs.shape.wavelengths.size

    XYZ_outer = colour.volume.solid_RoschMacAdam(
        cmfs=cmfs,
        illuminant=illuminant,
        point_order=point_order,
        filter_jagged_points=filter_jagged_points,
        bins=bins
    )

    R_white = colour.SpectralDistribution(
        data=np.ones_like(illuminant.values),
        domain=illuminant.wavelengths
    )
    Xw, Yw, Zw = colour.sd_to_XYZ(R_white, cmfs=cmfs, illuminant=illuminant)
    XYZ_rel = XYZ_outer / np.max(XYZ_outer[:, 1]) * Yw # Normalize to Y_white = 1 (perfect white diffuser)

    D65_xy = colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D65']
    with colour.domain_range_scale('reference'):
        Lab_ref = colour.XYZ_to_Lab(XYZ_rel, illuminant=D65_xy)

    a = Lab_ref[:, 1]
    b = Lab_ref[:, 2]
    chroma = np.hypot(a, b)
    hue_deg = (np.degrees(np.arctan2(b, a) + 360) % 360.0)

    rgb_phys = np.clip(colour.XYZ_to_sRGB(XYZ_rel, illuminant=D65_xy), 0.0, 1.0)

    Y_rel = XYZ_rel[:, 1]
    Y_nonzero = Y_rel[Y_rel > 0]
    gain = np.percentile(Y_nonzero, 99.0) if Y_nonzero.size else 1.0
    gain = gain if gain > 0 else 1.0
    XYZ_disp = XYZ_rel / gain
    rgb_bright = np.clip(
        colour.XYZ_to_sRGB(np.maximum(XYZ_disp, 0), illuminant=D65_xy),
        0.0, 1.0
    )

    debug = {
        'domain_global': colour.get_domain_range_scale(),
        'bins': bins,
        'point_order': point_order,
        'filter_jagged_points': filter_jagged_points,
        'Yw': float(Yw),
        'XYZ_outer_Y_minmax': (float(XYZ_outer[:, 1].min()), float(XYZ_outer[:, 1].max())),
        'XYZ_rel_Y_minmax': (float(XYZ_rel[:, 1].min()), float(XYZ_rel[:, 1].max())),
    }

    return XYZ_rel, Lab_ref, rgb_phys, rgb_bright, hue_deg, chroma, debug

def rgb_to_hex(rgb):
    rgb8 = (np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)
    return [f'#{r:02X}{g:02X}{b:02X}' for r, g, b in rgb8]