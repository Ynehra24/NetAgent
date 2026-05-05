import os
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from basefiles.logger import get_logger

log = get_logger(__name__)

# ── Signal Thresholds for Color Mapping ──
# These define the professional RF heatmap gradient
EXCELLENT_DBM = -50.0   # Green — strong signal
GOOD_DBM      = -60.0   # Yellow-green
FAIR_DBM      = -65.0   # Yellow-orange
WEAK_DBM      = -70.0   # Red — minimum usable

# Infrastructure device colors
ROUTER_COLOR     = (220, 50, 50, 255)       # Red
SWITCH_COLOR     = (50, 50, 220, 255)       # Blue
DATA_POINT_COLOR = (50, 150, 50, 255)       # Green

# AP dot colors per index
AP_DOT_COLORS = [
    (66, 133, 244, 255),
    (234, 67, 53, 255),
    (251, 188, 4, 255),
    (52, 168, 83, 255),
    (171, 71, 188, 255),
    (255, 112, 67, 255),
    (0, 172, 193, 255),
    (121, 85, 72, 255),
]


def _rssi_to_color(rssi: float) -> tuple:
    """
    Map an RSSI value (dBm) to an RGBA color on a professional gradient:
      ≥ -50 dBm → Green (excellent)
      -50 to -60 → Yellow-green (good)
      -60 to -65 → Yellow (fair)
      -65 to -70 → Orange-red (weak)
      < -70 dBm  → Red (no coverage)
    """
    if rssi >= EXCELLENT_DBM:
        return (0, 200, 0, 100)       # Green
    elif rssi >= GOOD_DBM:
        t = (rssi - GOOD_DBM) / (EXCELLENT_DBM - GOOD_DBM)
        r = int(180 * (1 - t))
        g = int(200 * t + 180 * (1 - t))
        return (r, g, 0, 90)
    elif rssi >= FAIR_DBM:
        t = (rssi - FAIR_DBM) / (GOOD_DBM - FAIR_DBM)
        r = int(220 * (1 - t) + 180 * t)
        g = int(180 * t + 140 * (1 - t))
        return (r, g, 0, 85)
    elif rssi >= WEAK_DBM:
        t = (rssi - WEAK_DBM) / (FAIR_DBM - WEAK_DBM)
        r = int(220)
        g = int(140 * t)
        return (r, g, 0, 80)
    else:
        return (200, 30, 30, 70)       # Red — dead zone


def _compute_signal_grid(placements: list, all_rooms: list, 
                         scale_factor: float, wall_attenuation_db: float,
                         eirp: float, width: int, height: int,
                         grid_resolution: int = 4) -> np.ndarray:
    """
    Compute RSSI at every pixel (sampled at grid_resolution intervals) 
    using FSPL + wall attenuation from each AP. Returns a 2D array of 
    best-case RSSI values.
    
    grid_resolution: compute every Nth pixel (4 = quarter resolution, then upscale)
    """
    grid_h = height // grid_resolution
    grid_w = width // grid_resolution
    
    # Initialize to very low signal
    signal_grid = np.full((grid_h, grid_w), -100.0, dtype=np.float32)
    
    if not placements:
        return signal_grid
    
    # Precompute AP positions in pixel space
    ap_positions = []
    for ap in placements:
        gx = ap["position"]["x"]
        gy = ap["position"]["y"]
        px = gx / 1000.0 * width
        py = gy / 1000.0 * height
        ap_positions.append((px, py, ap.get("placed_in_room", "")))
    
    # Precompute room bounding boxes in pixel space for wall crossing
    room_pixel_bbs = []
    for room in all_rooms:
        bb = room.get("bounding_box")
        if bb and len(bb) == 4:
            ymin, xmin, ymax, xmax = bb
            room_pixel_bbs.append({
                "name": room["name"],
                "px1": xmin / 1000.0 * width,
                "py1": ymin / 1000.0 * height,
                "px2": xmax / 1000.0 * width,
                "py2": ymax / 1000.0 * height,
            })

    # For each grid cell, find best signal from any AP
    for gy_idx in range(grid_h):
        py = (gy_idx + 0.5) * grid_resolution
        for gx_idx in range(grid_w):
            px = (gx_idx + 0.5) * grid_resolution
            
            best_rssi = -100.0
            
            for ap_px, ap_py, ap_room in ap_positions:
                # Distance in pixels → meters
                dist_px = math.sqrt((px - ap_px)**2 + (py - ap_py)**2)
                dist_m = (dist_px / width) * 1000.0 * scale_factor / 100.0
                
                if dist_m < 0.5:
                    rssi = eirp  # Right on top of AP
                else:
                    # FSPL at 5GHz (indoor adjusted)
                    fspl = 20 * math.log10(dist_m) + 20 * math.log10(5000) + 32.44 - 28
                    
                    # Quick wall crossing estimate: count how many room 
                    # boundaries the line from AP to this pixel crosses
                    walls = _fast_wall_count(ap_px, ap_py, px, py, 
                                            ap_room, room_pixel_bbs)
                    wall_penalty = walls * wall_attenuation_db
                    
                    rssi = eirp - fspl - wall_penalty
                
                if rssi > best_rssi:
                    best_rssi = rssi
            
            signal_grid[gy_idx, gx_idx] = best_rssi
    
    return signal_grid


def _fast_wall_count(ax: float, ay: float, bx: float, by: float,
                     ap_room: str, rooms: list) -> int:
    """
    Fast wall crossing estimate: sample 10 points along the line from AP to pixel,
    count how many distinct intermediate rooms the line passes through.
    Adjacent rooms = 1 wall, each intermediate room = +2 walls.
    """
    # Find which room the target pixel is in
    target_room = ""
    for room in rooms:
        if room["px1"] <= bx <= room["px2"] and room["py1"] <= by <= room["py2"]:
            target_room = room["name"]
            break
    
    if target_room == ap_room:
        return 0  # Same room, no walls
    
    # Sample 10 points along the line
    crossed = set()
    for i in range(1, 10):
        t = i / 10.0
        sx = ax + t * (bx - ax)
        sy = ay + t * (by - ay)
        for room in rooms:
            if room["name"] in (ap_room, target_room):
                continue
            if room["px1"] <= sx <= room["px2"] and room["py1"] <= sy <= room["py2"]:
                crossed.add(room["name"])
    
    return 1 + 2 * len(crossed)


def generate_heatmap(image_path: str, placement_plan: dict, 
                     output_path: str = "outputs/heatmap.png") -> str:
    """
    Professional RF coverage heatmap showing:
    - Continuous signal gradient (Green→Yellow→Red) computed from FSPL + wall attenuation
    - Grey zones for non-functional rooms (corridors, stairwells, bathrooms)
    - AP dots, Router, Switch, and Data Point markers
    - Professional legend with signal strength scale
    """
    log.info(f"=== Generating RF Coverage Heatmap ===")
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Original floor plan image not found: {image_path}")

    try:
        base_img = Image.open(image_path).convert("RGBA")
    except Exception as e:
        log.error(f"Failed to open image {image_path}: {e}")
        raise

    width, height = base_img.size
    log.info(f"Original image dimensions: {width}x{height}")

    # ── Extract RF parameters from placement plan ──
    placements = placement_plan.get("ap_placements", [])
    infra_devices = placement_plan.get("infra_devices", [])
    all_rooms = placement_plan.get("all_rooms", [])
    scale_factor = placement_plan.get("scale_factor_cm_per_unit", 4.0)
    wall_attenuation_db = placement_plan.get("wall_attenuation_db", 3.0)
    
    # Compute EIRP from tier specs
    eirp = 26.0 + 5.5  # default mid_tier: 26 dBm + 5.5 dBi
    
    # Load fonts
    try:
        font_label = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", max(11, int(width * 0.018)))
        font_legend = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", max(10, int(width * 0.015)))
        font_legend_title = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", max(12, int(width * 0.018)))
    except:
        font_label = ImageFont.load_default()
        font_legend = ImageFont.load_default()
        font_legend_title = ImageFont.load_default()

    dot_r = max(8, int(width * 0.014))
    small_r = max(5, int(width * 0.009))

    # ── STEP 1: Compute signal strength grid ──
    log.info(f"Computing RF signal grid ({width}x{height} at 1/4 resolution)...")
    signal_grid = _compute_signal_grid(
        placements, all_rooms, scale_factor, wall_attenuation_db,
        eirp, width, height, grid_resolution=4
    )
    log.info(f"Signal range: {signal_grid.min():.1f} to {signal_grid.max():.1f} dBm")

    # ── STEP 2: Render the gradient heatmap overlay ──
    grid_h, grid_w = signal_grid.shape
    heatmap_small = Image.new("RGBA", (grid_w, grid_h), (0, 0, 0, 0))
    
    for y in range(grid_h):
        for x in range(grid_w):
            rssi = signal_grid[y, x]
            color = _rssi_to_color(rssi)
            heatmap_small.putpixel((x, y), color)
    
    # Upscale to full resolution with smooth interpolation
    heatmap_overlay = heatmap_small.resize((width, height), Image.BILINEAR)
    # Apply slight blur for smooth gradient look
    heatmap_overlay = heatmap_overlay.filter(ImageFilter.GaussianBlur(radius=3))
    
    log.info("Rendered gradient heatmap overlay")

    # ── STEP 3: Draw grey zones for excluded rooms ──
    overlay = Image.new("RGBA", base_img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    
    # Build set of rooms that have APs
    ap_room_names = {ap["placed_in_room"] for ap in placements}
    
    GREY_ZONE_COLOR = (160, 160, 175, 50)
    for room in all_rooms:
        if room["name"] in ap_room_names:
            continue
        # Check if this room is covered by spillover from any AP
        is_covered = False
        for ap in placements:
            if room["name"] in ap.get("covers_rooms", []):
                is_covered = True
                break
        
        bb = room.get("bounding_box")
        if bb and len(bb) == 4:
            ymin, xmin, ymax, xmax = bb
            px1 = int((xmin / 1000.0) * width)
            py1 = int((ymin / 1000.0) * height)
            px2 = int((xmax / 1000.0) * width)
            py2 = int((ymax / 1000.0) * height)
            if not is_covered:
                draw.rectangle([px1, py1, px2, py2], fill=GREY_ZONE_COLOR)
                
                # Add a label in the center of the grey zone
                cx = (px1 + px2) // 2
                cy = (py1 + py2) // 2
                label_text = f"No AP\n({room['name']})"
                try:
                    # Make font slightly smaller for grey zones
                    grey_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", max(9, int(width * 0.012)))
                except:
                    grey_font = font_legend
                    
                bbox = draw.textbbox((0, 0), label_text, font=grey_font)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]
                draw.text((cx - tw//2, cy - th//2), label_text, fill=(80, 80, 80, 200), font=grey_font)
                
                log.info(f"Drew grey zone for excluded room '{room['name']}'")

    # ── STEP 4: Composite layers ──
    # Base image → heatmap gradient → grey zones → device markers
    composite = Image.alpha_composite(base_img, heatmap_overlay)
    composite = Image.alpha_composite(composite, overlay)
    
    # Create a new overlay for device markers
    marker_overlay = Image.new("RGBA", base_img.size, (255, 255, 255, 0))
    marker_draw = ImageDraw.Draw(marker_overlay)

    # ── STEP 5: Draw infrastructure devices ──
    for dev in infra_devices:
        gx = dev["position"]["x"]
        gy = dev["position"]["y"]
        px = int((gx / 1000.0) * width)
        py = int((gy / 1000.0) * height)
        dev_type = dev["type"]
        
        if dev_type == "Router":
            marker_draw.ellipse([px - dot_r, py - dot_r, px + dot_r, py + dot_r],
                         fill=ROUTER_COLOR, outline=(255, 255, 255, 255), width=2)
            _draw_label(marker_draw, px, py, dot_r, "Router", font_label, ROUTER_COLOR)
            log.info(f"Drew Router at pixel ({px}, {py})")
            
        elif dev_type == "Switch":
            marker_draw.rectangle([px - dot_r, py - dot_r, px + dot_r, py + dot_r],
                           fill=SWITCH_COLOR, outline=(255, 255, 255, 255), width=2)
            _draw_label(marker_draw, px, py, dot_r, "Switch", font_label, SWITCH_COLOR)
            log.info(f"Drew Switch at pixel ({px}, {py})")
            
        elif dev_type == "Data Point":
            diamond = [
                (px, py - small_r),
                (px + small_r, py),
                (px, py + small_r),
                (px - small_r, py),
            ]
            marker_draw.polygon(diamond, fill=DATA_POINT_COLOR, outline=(255, 255, 255, 255))
            _draw_label(marker_draw, px, py, small_r, dev["id"].upper().replace("_", ""), font_label, DATA_POINT_COLOR)
            log.info(f"Drew {dev['id']} at pixel ({px}, {py})")

    # ── STEP 6: Draw AP dots on top ──
    for idx, ap in enumerate(placements):
        color_idx = idx % len(AP_DOT_COLORS)
        dot_color = AP_DOT_COLORS[color_idx]
        
        gx = ap["position"]["x"]
        gy = ap["position"]["y"]
        px = int((gx / 1000.0) * width)
        py = int((gy / 1000.0) * height)
        
        # AP dot with white border
        marker_draw.ellipse([px - dot_r, py - dot_r, px + dot_r, py + dot_r],
                     fill=dot_color, outline=(255, 255, 255, 255), width=3)
        _draw_label(marker_draw, px, py, dot_r, f"AP{idx+1}", font_label, dot_color)
        log.info(f"Drawing AP{idx+1} dot at pixel ({px}, {py})")

    # Composite markers on top
    composite = Image.alpha_composite(composite, marker_overlay)

    # ── STEP 7: Create padded canvas with legend ──
    legend_entries = _build_legend_entries(placements, infra_devices)
    marker_size = max(6, dot_r // 2)
    padding_h = _calculate_legend_height(
        legend_entries, marker_size, width, font_legend_title, font_legend
    )

    canvas_w = width
    canvas_h = height + padding_h
    canvas = Image.new("RGBA", (canvas_w, canvas_h), (255, 255, 255, 255))
    canvas.paste(composite, (0, 0))

    legend_draw = ImageDraw.Draw(canvas)
    _draw_legend_bottom(legend_draw, width, height, padding_h, legend_entries,
                        marker_size, font_legend_title, font_legend)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    canvas.convert("RGB").save(output_path, format="PNG")
    
    log.info(f"=== Heatmap saved to {output_path} ===")
    return output_path


def _draw_label(draw, px, py, offset, text, font, color):
    """Draw a labeled tag next to a device marker."""
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    
    tx = px + offset + 5
    ty = py - th // 2
    
    draw.rectangle([tx - 2, ty - 1, tx + tw + 2, ty + th + 1], fill=(255, 255, 255, 210))
    draw.text((tx, ty), text, fill=(0, 0, 0, 255), font=font)


def _build_legend_entries(placements, infra_devices):
    """Collect all legend entries: (color, shape, label)."""
    entries = []
    routers = [d for d in infra_devices if d["type"] == "Router"]
    switches = [d for d in infra_devices if d["type"] == "Switch"]
    dps = [d for d in infra_devices if d["type"] == "Data Point"]

    if routers:
        entries.append((ROUTER_COLOR, "circle", f"Router (×{len(routers)})"))
    if switches:
        entries.append((SWITCH_COLOR, "square", f"Switch (×{len(switches)})"))
    if dps:
        entries.append((DATA_POINT_COLOR, "diamond", f"Data Point (×{len(dps)})"))
    if placements:
        entries.append((AP_DOT_COLORS[0], "circle", f"Access Point (×{len(placements)})"))

    # Signal strength gradient legend entries
    entries.append(((0, 200, 0, 255), "square", "Strong (≥-50dBm)"))
    entries.append(((200, 200, 0, 255), "square", "Good (-60dBm)"))
    entries.append(((220, 120, 0, 255), "square", "Weak (-70dBm)"))
    entries.append(((200, 30, 30, 255), "square", "No Signal"))
    entries.append(((160, 160, 175, 255), "square", "Excluded Zone"))
    return entries


def _entry_width(marker_size, label, font_body):
    """Calculate the pixel width of a single legend entry."""
    # Use a temporary image to measure text
    tmp = Image.new("RGBA", (1, 1))
    tmp_draw = ImageDraw.Draw(tmp)
    bbox = tmp_draw.textbbox((0, 0), label, font=font_body)
    label_w = bbox[2] - bbox[0]
    return marker_size * 2 + 6 + label_w + 20


def _calculate_legend_height(entries, marker_size, img_w, font_title, font_body):
    """Calculate how tall the legend area needs to be based on row wrapping."""
    # Measure title width
    tmp = Image.new("RGBA", (1, 1))
    tmp_draw = ImageDraw.Draw(tmp)
    title_bbox = tmp_draw.textbbox((0, 0), "Network Deployment", font=font_title)
    title_w = title_bbox[2] - title_bbox[0]

    row_height = max(24, marker_size * 2 + 10)
    x_start = 16 + title_w + 24
    max_x = img_w - 16  # right margin

    num_rows = 1
    x_cursor = x_start
    for _, _, label in entries:
        w = _entry_width(marker_size, label, font_body)
        if x_cursor + w > max_x and x_cursor > x_start:
            num_rows += 1
            x_cursor = 16  # new row starts at left margin (no title)
        x_cursor += w

    padding = 16 + num_rows * row_height + 16
    return max(60, padding)


def _draw_legend_bottom(draw, img_w, img_h, pad_h, entries, marker_size,
                        font_title, font_body):
    """Draw a wrapping legend bar in the white padding strip below the image."""
    row_height = max(24, marker_size * 2 + 10)

    title = "Network Deployment"
    title_bbox = draw.textbbox((0, 0), title, font=font_title)
    title_w = title_bbox[2] - title_bbox[0]
    title_h = title_bbox[3] - title_bbox[1]

    draw.line([(0, img_h), (img_w, img_h)], fill=(200, 200, 200, 255), width=1)

    # First row y-center
    first_row_y = img_h + 16 + row_height // 2

    # Draw title on the first row
    draw.text((16, first_row_y - title_h // 2), title, fill=(30, 30, 30, 255), font=font_title)

    x_start_first_row = 16 + title_w + 24
    x_start_next_rows = 16
    max_x = img_w - 16

    x_cursor = x_start_first_row
    current_row = 0
    cy = first_row_y

    for color, shape, label in entries:
        w = _entry_width(marker_size, label, font_body)

        # Wrap to next row if this entry would overflow
        x_start = x_start_first_row if current_row == 0 else x_start_next_rows
        if x_cursor + w > max_x and x_cursor > x_start:
            current_row += 1
            x_cursor = x_start_next_rows
            cy = first_row_y + current_row * row_height

        ex = x_cursor + marker_size

        if shape == "circle":
            draw.ellipse([ex - marker_size, cy - marker_size, ex + marker_size, cy + marker_size],
                         fill=color, outline=(80, 80, 80, 255), width=1)
        elif shape == "square":
            draw.rectangle([ex - marker_size, cy - marker_size, ex + marker_size, cy + marker_size],
                           fill=color, outline=(80, 80, 80, 255), width=1)
        elif shape == "diamond":
            diamond = [(ex, cy - marker_size), (ex + marker_size, cy),
                       (ex, cy + marker_size), (ex - marker_size, cy)]
            draw.polygon(diamond, fill=color, outline=(80, 80, 80, 255))

        label_bbox = draw.textbbox((0, 0), label, font=font_body)
        label_h = label_bbox[3] - label_bbox[1]
        label_w = label_bbox[2] - label_bbox[0]
        draw.text((ex + marker_size + 6, cy - label_h // 2), label, fill=(30, 30, 30, 255), font=font_body)

        x_cursor += w
