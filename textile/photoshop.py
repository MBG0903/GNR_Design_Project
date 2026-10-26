"""Validate agent polygons and build an actual Photoshop JSX edit script.

The script operates on a duplicate of an explicitly selected raster, never on
an arbitrary active document. Execution requires desktop Photoshop.
"""
import json
from pathlib import Path


def validate_plan(plan, size):
    if not isinstance(plan,dict) or set(plan)!={'findings','unresolved','operations'}:
        raise ValueError('Plan must contain findings, unresolved, operations only.')
    if not isinstance(plan['findings'],list) or not isinstance(plan['unresolved'],list):
        raise ValueError('Invalid findings/uncertainty list.')
    if not all(isinstance(x,str) for x in plan['unresolved']):
        raise ValueError('Unresolved issues must be text.')
    ops=plan['operations']
    if not isinstance(ops,list) or len(ops)>20:
        raise ValueError('At most 20 polygon edits per iteration.')
    w,h=size
    for op in ops:
        if not isinstance(op,dict) or set(op)!={'rule','reason','colour','points'}:
            raise ValueError('Unexpected operation format.')
        if op['rule'] not in {f'R{i}' for i in range(1,17)} or not isinstance(op['reason'],str) or not op['reason'].strip():
            raise ValueError('Each edit needs a rule and explanation.')
        if op['colour'] not in ['red','white']:
            raise ValueError('Only red and white fills are allowed.')
        points=op['points']
        if not isinstance(points,list) or not 3<=len(points)<=100:
            raise ValueError('A polygon needs 3–100 vertices.')
        for point in points:
            if not isinstance(point,list) or len(point)!=2 or any(type(v) is not int for v in point):
                raise ValueError('Native pixel coordinates must be integer pairs.')
            if not (0<=point[0]<=w and 0<=point[1]<=h):
                raise ValueError('Polygon outside native canvas.')
        area=abs(sum(points[i][0]*points[(i+1)%len(points)][1]-points[(i+1)%len(points)][0]*points[i][1]
                     for i in range(len(points))))/2
        if area<1 or area>w*h*0.05:
            raise ValueError('Polygon must cover 1 pixel to 5% of canvas; larger changes need designer review.')
    return plan


def write_jsx(plan, size, destination, current_image):
    validate_plan(plan,size)
    w,h=size
    raw = Path(current_image).read_bytes()
    checksum = 0
    for byte in raw:
        checksum = (checksum * 31 + byte) & 0xffffffff
    # All model content is serialized as JSON data; never interpolate executable code.
    data=json.dumps(plan['operations'],ensure_ascii=True)
    script='''#target photoshop
(function () {
    var ops = __OPS__;
    var source = File.openDialog("Select the exact current PNG reviewed by the agent", "*.png");
    if (!source) return;
    source.encoding = "BINARY";
    if (!source.open("r")) throw new Error("Cannot read selected PNG.");
    var bytes = source.read(); source.close();
    var checksum = 0;
    for (var b=0; b<bytes.length; b++) checksum = (checksum * 31 + bytes.charCodeAt(b)) >>> 0;
    if (bytes.length !== __LENGTH__ || checksum !== __CHECKSUM__) {
        alert("Selected file differs from the reviewed PNG. No edits were applied.");
        return;
    }
    var originalUnits = app.preferences.rulerUnits;
    var document = null;
    app.preferences.rulerUnits = Units.PIXELS;
    try {
        var input = app.open(source);
        if (input.width.as("px") !== __WIDTH__ || input.height.as("px") !== __HEIGHT__) {
            throw new Error("Wrong dimensions. Select the reviewed native-size PNG.");
        }
        document = input.duplicate("Textile agent corrected");
        app.activeDocument = document;
        document.changeMode(ChangeMode.RGB);
        document.bitsPerChannel = BitsPerChannelType.EIGHT;
        document.artLayers.add().name = "Agent corrections";
        for (var i=0; i<ops.length; i++) {
            var op = ops[i];
            document.selection.select(op.points, SelectionType.REPLACE, 0, false);
            var colour = new SolidColor();
            colour.rgb.red = 255;
            colour.rgb.green = op.colour === "red" ? 0 : 255;
            colour.rgb.blue = op.colour === "red" ? 0 : 255;
            document.selection.fill(colour, ColorBlendMode.NORMAL, 100, false);
            document.selection.deselect();
        }
        var folder = Folder.selectDialog("Select a NEW output folder for this iteration");
        if (!folder) throw new Error("Save cancelled; duplicate remains open for inspection.");
        var psd = File(folder.fsName + "/agent_corrected.psd");
        var png = File(folder.fsName + "/agent_corrected.png");
        if (psd.exists || png.exists) throw new Error("Output exists; choose a new iteration folder.");
        document.saveAs(psd, new PhotoshopSaveOptions(), true, Extension.LOWERCASE);
        document.saveAs(png, new PNGSaveOptions(), true, Extension.LOWERCASE);
        alert("Agent edits saved. Revalidate the PNG and request the next review. No production approval is implied.");
    } catch (error) {
        alert("Textile agent: " + error.message);
    } finally {
        app.preferences.rulerUnits = originalUnits;
    }
})();
'''.replace('__OPS__',data).replace('__WIDTH__',str(w)).replace('__HEIGHT__',str(h)).replace('__LENGTH__',str(len(raw))).replace('__CHECKSUM__',str(checksum))
    Path(destination).write_text(script)
    return str(destination)
