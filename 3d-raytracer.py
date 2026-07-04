import math
import numpy as np
from numba import njit, prange, njit, float64
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import time
import OpenEXR
import Imath
import threading

# Resolution steps for interactive rendering
low_res = 200
high_res = 800
render_in_progress = False

# Globals for HDR data
pano_data = None
pano_width = 0
pano_height = 0

def load_panorama():
    """Loads OpenEXR background."""
    global pano_data, pano_width, pano_height
    try:
        exr_file = OpenEXR.InputFile("background/pano.exr")
        header = exr_file.header()
        dw = header['dataWindow']
        pano_width = dw.max.x - dw.min.x + 1
        pano_height = dw.max.y - dw.min.y + 1

        hdr_data = Imath.PixelType(Imath.PixelType.FLOAT)

        r_str = exr_file.channel('R', hdr_data)
        g_str = exr_file.channel('G', hdr_data)
        b_str = exr_file.channel('B', hdr_data)

        r = np.frombuffer(r_str, dtype=np.float32).reshape(pano_height, pano_width)
        g = np.frombuffer(g_str, dtype=np.float32).reshape(pano_height, pano_width)
        b = np.frombuffer(b_str, dtype=np.float32).reshape(pano_height, pano_width)

        pano_data = np.stack([r, g, b], axis=2)
        print(f"Loaded panorama: {pano_width}x{pano_height}")
        return True

    except Exception as e:
        print(f"Error loading panorama: {e}")
        pano_width, pano_height = 2048, 1024
        altitude = np.linspace(1.0, 0.0, pano_height, dtype=np.float32)[:, np.newaxis, np.newaxis]
        sky_color = np.array([0.5, 0.7, 1.0], dtype=np.float32)
        pano_data = altitude * sky_color * np.ones((1, pano_width, 1), dtype=np.float32)
        print("Using fallback gradient sky")
        return False


@njit(cache=True, fastmath=True, inline='always')
def tonemap_reinhard(color, exposure):
    """Tonemap image using exposure value"""
    c = color * exposure
    result = np.empty(3)
    result[0] = c[0] / (1.0 + c[0])
    result[1] = c[1] / (1.0 + c[1])
    result[2] = c[2] / (1.0 + c[2])
    return result


@njit(cache=True, fastmath=True)
def sample_equirectangular(direction, pano_array, width, height):
    """Get panorama color based on direction vector"""
    d = direction / ( np.linalg.norm(direction) + 1e-12 )

    # horizontal angle
    phi = np.arctan2(d[0], d[2])

    d_y = d[1]
    if d_y < -1.0:
        d_y = -1.0
    elif d_y > 1.0:
        d_y = 1.0
    theta = np.arccos(d_y)

    u = (phi + np.pi) / (2.0 * np.pi)
    v = theta / np.pi

    x = u * (width - 1)
    y = v * (height - 1)

    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = min(x0 + 1, width - 1)
    y1 = min(y0 + 1, height - 1)

    fx = x - x0
    fy = y - y0

    c00 = pano_array[y0, x0]
    c10 = pano_array[y0, x1]
    c01 = pano_array[y1, x0]
    c11 = pano_array[y1, x1]

    c0 = c00 * (1 - fx) + c10 * fx
    c1 = c01 * (1 - fx) + c11 * fx
    return c0 * (1 - fy) + c1 * fy


@njit(cache=True, fastmath=True, inline='always')
def checker(x, y, scale, c0=0, c1=1):
    """2D square checkerboard pattern"""
    if scale<=0:
        return 1.0
    checkerx = int(x / scale) % 2
    checkery = int(y / scale) % 2
    if x < 0: checkerx = 1 - checkerx
    if y < 0: checkery = 1 - checkery
    return c0 if checkerx == checkery else c1


@njit(cache=True, fastmath=True, inline='always')
def reflection(vector, normal):
    """Reflection about normal"""
    return vector - 2.0 * np.dot(vector, normal) * normal


@njit(cache=True, fastmath=True, inline='always')
def lambertian_reflection(vector, normal, roughness=0.1):
    """
    Scatters a vector based on a roughness parameter.
    roughness = 0.0 -> Fully reflective (Specular)
    roughness = 1.0 -> Fully scattered (Lambertian)
    """
    # 1. Calculate the perfect reflection vector
    # (Note: Assumes incoming 'vector' points toward the surface)
    reflect_dir = vector - 2.0 * np.dot(vector, normal) * normal

    # 2. Generate a random point on a unit sphere (Lambertian target)
    theta = 2.0 * np.pi * np.random.random()
    phi = np.arccos(2.0 * np.random.random() - 1.0)
    sin_phi = np.sin(phi)

    random_on_sphere = np.array([
        sin_phi * np.cos(theta),
        sin_phi * np.sin(theta),
        np.cos(phi)
    ])

    # 3. Choose the target based on roughness
    # If roughness is 0, we use pure reflection.
    # If roughness is 1, we offset the normal (pure Lambertian).
    if roughness > 0.0:
        # Interpolate the base direction between pure reflection and the normal
        base_dir = (1.0 - roughness) * reflect_dir + roughness * normal
        scatter_direction = base_dir + roughness * random_on_sphere
    else:
        scatter_direction = reflect_dir

    # 4. Catch degenerate cases where the vector sums to zero
    if np.dot(scatter_direction, scatter_direction) < 1e-16:
        return normal

    # 5. Normalize and return
    return scatter_direction / np.linalg.norm(scatter_direction)


@njit(cache=True, fastmath=True, inline='always')
def refraction(vector, normal, n1, n2):
    """Snell's Law refraction n1 * sin(theta1) = n2 * sin(theta2)"""
    cos_theta_i = -np.dot(vector, normal)
    if cos_theta_i > 1.0:
        return np.zeros(3), False
    sin_theta_i = np.sqrt(1.0 - cos_theta_i * cos_theta_i)
    sin_theta_t = (n1 / n2) * sin_theta_i
    if sin_theta_t > 1.0:
        return np.zeros(3), False
    cos_theta_t = np.sqrt(1.0 - sin_theta_t * sin_theta_t)
    n_ratio = n1 / n2
    refr = n_ratio * vector + (n_ratio * cos_theta_i - cos_theta_t) * normal
    return refr, True


@njit(cache=True, fastmath=True, inline='always')
def schlick_fresnel(cos_theta_i, n1, n2):
    """
    Schlick approximation to Fresnel reflection and transmission.
    """
    # Clamp cosine to avoid floating-point errors
    cos_i = max(0.0, min(1.0, cos_theta_i))

    # Compute base reflectance at normal incidence (R0)
    r0 = ((n1 - n2) / (n1 + n2)) ** 2

    # Internal Reflection
    if n1 > n2:
        sin_theta_i = np.sqrt(1.0 - cos_i * cos_i)
        sin_theta_t = (n1 / n2) * sin_theta_i

        # Total Internal Reflection (TIR)
        if sin_theta_t >= 1.0:
            return 1.0

        # Schlick cosine approximation
        cos_t = np.sqrt(1.0 - sin_theta_t * sin_theta_t)
        cos_theta = max(0.0, min(1.0, cos_t))
    else:
        # Rare -> Dense medium transition (Standard External Reflection)
        cos_theta = cos_i

    # Schlick power formula
    return r0 + (1.0 - r0) * ((1.0 - cos_theta) ** 5)


@njit(cache=True, fastmath=True, inline='always')
def ray_plane_intersection(origin, direction, plane_origin, plane_normal):
    """Calculate intersection of ray and plane"""
    dp = np.dot(direction, plane_normal)
    if abs(dp) < 1e-10:
        return np.zeros(3), False
    d = np.dot(plane_origin - origin, plane_normal) / dp
    if d > 0:
        return origin + d * direction, True
    return np.zeros(3), False


@njit(cache=True, fastmath=True, inline='always')
def ray_sphere_intersection(origin, direction, sphere_center, sphere_radius, n1, n2):
    """Calculate intersection of ray and sphere"""
    oc = origin - sphere_center
    a = np.dot(direction, direction)
    h = -np.dot(direction, oc)
    c = np.dot(oc, oc) - sphere_radius * sphere_radius
    discriminant = h * h - a * c

    dist2 = np.dot(oc, oc)
    is_inside = dist2 < sphere_radius * sphere_radius + 1e-10
    eps_point = oc + direction * 1e-5
    is_directed_outward = np.dot(eps_point, eps_point) > sphere_radius * sphere_radius

    if discriminant < 0.0:
        return np.zeros(3), False, is_inside, is_directed_outward, np.zeros(3), n1, n2

    sqrt_disc = np.sqrt(discriminant)
    t = (h - sqrt_disc) / a
    if t < 1e-4:
        t = (h + sqrt_disc) / a
        if t < 1e-4:
            return np.zeros(3), False, is_inside, is_directed_outward, np.zeros(3), n1, n2

    ray_end = origin + t * direction
    normal = ray_end - sphere_center
    inv_len = 1.0 / np.sqrt(np.dot(normal, normal))
    normal *= inv_len

    # Determine
    if is_inside:
        # Ray is inside the sphere and exiting
        normal = -normal
        ni, nt = n2, n1
    else:
        # Ray is outside the sphere and entering
        ni, nt = n1, n2

    return ray_end, True, is_inside, is_directed_outward, normal, ni, nt


@njit(cache=True, fastmath=True, inline='always')
def get_ior(wavelength, cauchyA, cauchyB):
    """Calculate IOR via Cauchy dispersion formula"""
    return cauchyA + cauchyB * 1.0e6 / (wavelength * wavelength)


@njit(cache=True, fastmath=True)
def trace_rays(origin, direction, intensity, max_depth, n1, n2, color,
                       sphere_center, sphere_radius, pano_array, pano_width, pano_height,
                       background_center, background_normal, checker_scale,
                       exposure, checker_intensity, dispersion, bg_intensity):
    """Trace ray from pixel location and return resulting color"""
    max_stack = max_depth * 4
    origins = np.empty((max_stack, 3))
    directions = np.empty((max_stack, 3))
    intensities = np.empty(max_stack)
    depths = np.empty(max_stack, dtype=np.int32)

    origins[0] = origin
    directions[0] = direction
    intensities[0] = intensity
    depths[0] = 0
    stack_size = 1

    result_color = np.zeros(3)

    while stack_size > 0:
        stack_size -= 1
        ray_origin = origins[stack_size]
        ray_direction = directions[stack_size]
        ray_intensity = intensities[stack_size]
        depth = depths[stack_size]

        if depth > max_depth:
            continue

        # Sphere intersection
        (sphere_hit_point, has_sphere, is_inside, is_directed_outward,
         sphere_normal, ni, nt) = ray_sphere_intersection(
            ray_origin, ray_direction,
            sphere_center, sphere_radius,
            n1, n2
        )

        # Plane intersection
        plane_hit_point, has_plane = ray_plane_intersection(
            ray_origin, ray_direction,
            background_center, background_normal
        )

        # distances along ray (squared)
        sphere_dist2 = 1e30

        if has_sphere:
            dx = sphere_hit_point[0] - ray_origin[0]
            dy = sphere_hit_point[1] - ray_origin[1]
            dz = sphere_hit_point[2] - ray_origin[2]
            sphere_dist2 = dx*dx + dy*dy + dz*dz

        if has_plane:
            dxp = plane_hit_point[0] - ray_origin[0]
            dyp = plane_hit_point[1] - ray_origin[1]
            dzp = plane_hit_point[2] - ray_origin[2]
            plane_dist2 = dxp*dxp + dyp*dyp + dzp*dzp

            # If plane is closer, shade plane
            if has_plane and (not has_sphere or plane_dist2 < sphere_dist2):
                if checker_intensity > 0.0:
                    if -25.4 < plane_hit_point[0] < 25.4 and -25.4 < plane_hit_point[2] < 25.4:
                        pattern = checker(plane_hit_point[0], plane_hit_point[2], checker_scale, c0=0.1, c1=1.0)
                        result_color += ray_intensity * pattern * checker_intensity * color

                        # Calculate angle relative to the surface normal
                        cos_theta = -np.dot(ray_direction, background_normal)

                        # Calculate reflection intensity from Schlick approximation
                        fresnel_reflectance = schlick_fresnel(cos_theta, 1.0, 1.5)

                        # Propagate reflection
                        if fresnel_reflectance > 0.0 and stack_size < max_stack - 1:
                            v_refl = reflection(ray_direction, background_normal)
                            origins[stack_size] = plane_hit_point + background_normal * 1e-4
                            directions[stack_size] = v_refl
                            intensities[stack_size] = ray_intensity * max(pattern, 0.1) * fresnel_reflectance
                            depths[stack_size] = depth + 1
                            stack_size += 1

                        continue

        # If sphere is closer, shade sphere
        if has_sphere and not (is_inside and is_directed_outward):
            cos_theta = -(ray_direction[0]*sphere_normal[0] +
                          ray_direction[1]*sphere_normal[1] +
                          ray_direction[2]*sphere_normal[2])

            fresnel_reflectance = schlick_fresnel(cos_theta, ni, nt)
            fresnel_transmission = 1.0 - fresnel_reflectance

            v_refr, has_refr = refraction(ray_direction, sphere_normal, ni, nt)
            v_refl = reflection(ray_direction, sphere_normal)

            # Propagate refraction
            if has_refr and stack_size < max_stack - 1:
                origins[stack_size] = sphere_hit_point
                directions[stack_size] = v_refr
                intensities[stack_size] = ray_intensity * fresnel_transmission
                depths[stack_size] = depth + 1
                stack_size += 1

            # Propagate reflection
            if stack_size < max_stack - 1:
                origins[stack_size] = sphere_hit_point
                directions[stack_size] = v_refl
                intensities[stack_size] = ray_intensity * fresnel_reflectance
                depths[stack_size] = depth + 1
                stack_size += 1

            continue

        # Background
        sky_color = sample_equirectangular(ray_direction, pano_array, pano_width, pano_height)
        result_color += ray_intensity * sky_color * color * bg_intensity
        continue

    return result_color


@njit(parallel=True, cache=True, fastmath=True)
def render_parallel(img_size, max_depth, intensity_val, index_medium, index_bubble,
                    camera_origin, camera_lookat, fov, sphere_center, sphere_radius,
                    pano_array, pano_width, pano_height, exposure, background_center,
                    background_normal, checker_scale,
                    tonemap, checker_intensity,
                    dispersion, bg_intensity):
    """Render rows in parallel across CPU cores"""
    image = np.zeros((img_size, img_size, 3))

    w = camera_lookat - camera_origin
    focal_length = np.linalg.norm(w)
    up = np.array([0.0, 1.0, 0.0])
    w = w / focal_length

    u = np.cross(up, w)
    u = u / np.linalg.norm(u)
    v = np.cross(w, u)
    v = v / np.linalg.norm(v)

    fov_rad = fov * np.pi / 180.0
    image_height = focal_length * np.tan(fov_rad / 2.0) * 2.0
    pixel_size = image_height / float(img_size)
    upper_left = camera_lookat - camera_origin - u * image_height / 2.0 + v * image_height / 2.0

    if dispersion > 0:
        wavelengths = np.array([440, 555, 650], dtype=np.float64)
        colors = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.float64)
    else:
        wavelengths = np.array([555], dtype=np.float64)
        colors = np.array([[1, 1, 1]], dtype=np.float64)

    n1arr = np.empty(len(wavelengths))
    n2arr = np.empty(len(wavelengths))
    for i in range(len(wavelengths)):
        n1arr[i] = get_ior(wavelengths[i], index_medium, -dispersion)
        n2arr[i] = get_ior(wavelengths[i], index_bubble, -dispersion)

    for iy in prange(img_size):
        for ix in range(img_size):
            for i in range(len(wavelengths)):
                color = colors[i]
                n1 = n1arr[i]
                n2 = n2arr[i]

                pixel_center = upper_left + u * pixel_size * ix - v * pixel_size * iy
                direction = pixel_center - camera_origin
                direction = direction / np.linalg.norm(direction)

                # trace ray from pixel location and get resulting color
                pixel_color = trace_rays(
                    camera_origin, direction, intensity_val, max_depth, n1, n2, color,
                    sphere_center, sphere_radius, pano_array, pano_width, pano_height,
                    background_center, background_normal, checker_scale,
                    exposure, checker_intensity, dispersion, bg_intensity
                )

                if tonemap:
                    pixel_color = tonemap_reinhard(pixel_color, 1.0)

                image[iy, ix] += pixel_color

    return image


class InteractiveBubbleRaytracer:
    def __init__(self):
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(12, 8), facecolor='#1a1a1a')
        self.fig.patch.set_facecolor('#1a1a1a')
        self.ax_img = plt.axes([0.0, 0.15, 0.6, 0.7], facecolor='#0a0a0a')
        self.ax_img.set_xticks([])
        self.ax_img.set_yticks([])

        # default parameters
        self.defaults = {
            'camera_distance': 4,
            'camera_theta': 0.0,
            'camera_phi': 90.0,
            'fov': 90,
            'max_depth': 5,
            'intensity': 1.0,
            'index_medium': 1.33,
            'index_bubble': 1.0,
            'sphere_radius': 1.0,
            'exposure': 2.0,  # Static lock value
            'checker_scale': 1,
            'checker_intensity': 1.0,
            'bg_intensity': 5.0,
            'tonemap': True,
            'dispersion': 0.0
        }

        self.params = self.defaults.copy()

        self.image_obj = None
        self.render_time = 0
        self.is_interactive = False
        self.high_res_timer = None
        self.pending_high_res = False

        self.create_sliders()

        self.dragging = False
        self.last_pos = None
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)

        self.update(None, force_high_res=True)

    def create_sliders(self):
        """Construct interface sliders"""
        slider_x = 0.68
        slider_y = 0.83
        slider_width = 0.25
        slider_height = 0.018
        slider_spacing = 0.06

        def build_slider(label, amin, amax, param_key, step=None):
            """Helper function to build a slider with spacing"""
            nonlocal slider_y
            ax = plt.axes([slider_x, slider_y, slider_width, slider_height], facecolor='#2a2a2a')
            slider = Slider(ax, label, amin, amax, valinit=self.params[param_key], valstep=step, color='green')
            slider.label.set_color('white')
            slider.valtext.set_color('white')
            slider.on_changed(self.on_slider_change)
            slider_y -= slider_spacing
            return slider

        self.distance_slider = build_slider('Camera Dist', 0.25, 50, 'camera_distance')
        self.fov_slider = build_slider('FOV', 1, 135, 'fov')
        self.depth_slider = build_slider('Max Depth', 0, 10, 'max_depth', step=1)
        self.radius_slider = build_slider('Bubble Radius', 0.1, 10, 'sphere_radius')
        self.index_medium_slider = build_slider('Index Medium', 1.0, 2.0, 'index_medium')
        self.index_bubble_slider = build_slider('Index Bubble', 1.0, 2.0, 'index_bubble')
        self.checker_slider = build_slider('Checker Scale', 0.0, 10, 'checker_scale', step=0.1)
        self.checker_intensity_slider = build_slider('Checker Intensity', 0.0, 10.0, 'checker_intensity')
        self.bg_intensity_slider = build_slider('BG Intensity', 0.0, 100.0, 'bg_intensity')
        self.tonemap_slider = build_slider('Tone Mapping', 0, 1, 'tonemap', step=1)
        self.dispersion_slider = build_slider('Dispersion', 0, 0.1, 'dispersion', step=0.001)

        # Button
        ax_button = plt.axes([0.75, 0.1, 0.1, 0.05], facecolor='#555555')
        self.reset_button = Button(ax_button, 'Reset', color='#555555', hovercolor='#999999')
        self.reset_button.label.set_color('white')
        self.reset_button.on_clicked(self.reset)
        self.info_text = self.fig.text(0.0, 0.0, 'Reset Sliders', fontsize=10, color='white')

    def get_camera_position(self):
        """Get Cartesian coordinates of camera"""
        theta = -np.deg2rad(self.params['camera_theta'])
        phi = np.deg2rad(self.params['camera_phi'])
        distance = self.params['camera_distance']
        return distance * np.array([np.sin(phi) * np.sin(theta), np.cos(phi), np.sin(phi) * np.cos(theta)])

    def on_slider_change(self, val):
        """Slider change handler"""
        self.is_interactive = True
        self.update(None, force_high_res=False) # low res update
        self.is_interactive = False
        self.schedule_high_res_render()

    def schedule_high_res_render(self):
        """Schedule high res render after user interaction stops"""
        delay_time = 0.5 # delay before switching to high res mode
        if self.high_res_timer is not None:
            self.high_res_timer.cancel()
        self.pending_high_res = True
        self.high_res_timer = threading.Timer(delay_time, self.trigger_high_res_render)
        self.high_res_timer.start()

    def trigger_high_res_render(self):
        """Trigger high res render"""
        if self.pending_high_res:
            self.pending_high_res = False
            self.update(None, force_high_res=True) # high res update

    def update(self, val, force_high_res=False):
        """Main update loop"""
        global render_in_progress, pano_data, pano_width, pano_height
        if render_in_progress:
            return
        render_in_progress = True

        # dynamic resolution
        img_size = low_res if (self.is_interactive and not force_high_res) else high_res
        res_label = "LOW" if img_size == low_res else "HIGH"

        self.params['camera_distance'] = self.distance_slider.val
        self.params['fov'] = self.fov_slider.val
        self.params['max_depth'] = int(self.depth_slider.val)
        self.params['sphere_radius'] = self.radius_slider.val
        self.params['index_medium'] = self.index_medium_slider.val
        self.params['index_bubble'] = self.index_bubble_slider.val
        self.params['checker_scale'] = self.checker_slider.val
        self.params['checker_intensity'] = self.checker_intensity_slider.val
        self.params['bg_intensity'] = self.bg_intensity_slider.val
        self.params['tonemap'] = bool(self.tonemap_slider.val)
        self.params['dispersion'] = self.dispersion_slider.val

        camera_origin = self.get_camera_position()
        camera_lookat = np.array([0.0, 0.0, 0.0])
        sphere_center = np.array([0.00001, 0.0, 0.0]) # offset to avoid singularity

        background_center = np.array([0.0, -1.0, 0.0])
        background_normal = np.array([0.0, 1.0, 0.0])

        start_time = time.time()
        result = render_parallel(
            img_size, self.params['max_depth'], self.params['intensity'],
            self.params['index_medium'], self.params['index_bubble'],
            camera_origin, camera_lookat, self.params['fov'], sphere_center,
            self.params['sphere_radius'], pano_data, pano_width, pano_height,
            self.params['exposure'], background_center,
            background_normal, self.params['checker_scale'],
            self.params['tonemap'],
            self.params['checker_intensity'], self.params['dispersion'], self.params['bg_intensity']
        )
        self.render_time = time.time() - start_time
        result = np.clip(result, 0, 1)

        if self.image_obj is None:
            self.image_obj = self.ax_img.imshow(result)
        else:
            self.image_obj.set_data(result)

        info_str = f"Render time: {self.render_time:.2f}s | Resolution: {img_size}x{img_size} ({res_label}) | θ={self.params['camera_theta']:.1f}° φ={self.params['camera_phi']:.1f}°"
        self.info_text.set_text(info_str)

        self.fig.canvas.draw_idle()
        render_in_progress = False

    def on_press(self, event):
        """Mouse button press handler"""
        if event.inaxes == self.ax_img:
            self.dragging = True
            self.is_interactive = True
            self.last_pos = (event.xdata, event.ydata)
            if self.high_res_timer is not None:
                self.high_res_timer.cancel()
                self.pending_high_res = False

    def on_release(self, event):
        """Mouse button release handler"""
        self.dragging = False
        self.last_pos = None
        self.is_interactive = False
        self.schedule_high_res_render()

    def on_motion(self, event):
        """Mouse motion handler"""
        if self.dragging and event.inaxes == self.ax_img and self.last_pos is not None:
            dx = event.xdata - self.last_pos[0]
            dy = event.ydata - self.last_pos[1]
            rotation_speed = 0.5

            self.params['camera_theta'] += dx * rotation_speed
            new_phi = self.params['camera_phi'] + dy * rotation_speed
            self.params['camera_phi'] = np.clip(new_phi, 0.01, 179.0)

            self.update(None, force_high_res=False)
            self.last_pos = (event.xdata, event.ydata)

    def on_scroll(self, event):
        """Mouse scroll handler"""
        if event.inaxes == self.ax_img:
            self.is_interactive = True
            zoom_speed = 0.1
            current_distance = self.params['camera_distance']

            if event.button == 'up':
                new_distance = current_distance * (1.0 - zoom_speed)
            elif event.button == 'down':
                new_distance = current_distance * (1.0 + zoom_speed)
            else:
                return

            new_distance = np.clip(new_distance, 0.1, 75.0)
            self.params['camera_distance'] = new_distance
            self.distance_slider.set_val(new_distance)
            self.is_interactive = False
            self.schedule_high_res_render()

    def reset(self, event):
        """Reset button reset function"""
        self.params['camera_theta'] = self.defaults['camera_theta']
        self.params['camera_phi'] = self.defaults['camera_phi']

        self.distance_slider.set_val(self.defaults['camera_distance'])
        self.fov_slider.set_val(self.defaults['fov'])
        self.depth_slider.set_val(self.defaults['max_depth'])
        self.radius_slider.set_val(self.defaults['sphere_radius'])
        self.index_medium_slider.set_val(self.defaults['index_medium'])
        self.index_bubble_slider.set_val(self.defaults['index_bubble'])
        self.checker_slider.set_val(self.defaults['checker_scale'])
        self.checker_intensity_slider.set_val(self.defaults['checker_intensity'])
        self.bg_intensity_slider.set_val(self.defaults['bg_intensity'])
        self.tonemap_slider.set_val(int(self.defaults['tonemap']))
        self.dispersion_slider.set_val(self.defaults['dispersion'])

    def show(self):
        plt.show()


if __name__ == '__main__':
    load_panorama()
    app = InteractiveBubbleRaytracer()
    app.show()
