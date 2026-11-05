# CELL 10: FINAL — conservative pass from the ORIGINAL Photoshop draft.
# Independent of intermediate RED_FINAL/RED_SEP; never uses designer truth to select edits.
V5_CORE.require_input_context(globals())
V5_CORE.install_bindings(globals())

proposal = None
if V5_USE_LEARNED_FINISHER:
    if globals().get('V5_LEARNED_CONTEXT') != V5_ACTIVE_INPUT or 'RED_LEARN' not in globals():
        raise ValueError('Run Cell 9 for this design before enabling its learned proposal.')
    proposal = RED_LEARN

RED_DONE, V5_LAST_FINISHING_REPORT = V5_CORE.finish_from_draft(
    RED_PS, learned_proposal=proposal,
    max_shift=globals().get('MAX_SHIFT', 2),
    max_repeat=globals().get('MAX_REPEAT', 8),
    max_edits=globals().get('V5_MAX_CORNER_EDITS', 250),
    max_changed_fraction=V5_MAX_CHANGED_FRACTION,
    protected_white=globals().get('V5_PROTECTED_WHITE'))
V5_CORE.record_stage(globals(), 'FINAL: Photoshop plus bounded staircase edits', RED_DONE)
bmp, png, pure_ok = save_production(RED_DONE, DESIGN)
Image.fromarray(rgb_of(RED_DONE)).save(DESIGN_DIR / '6 final conservative staircase.png')

# Separate diagnostic image: blue = red removed, orange = red added.
# These colours are never saved in the Step_3 output.
difference = np.full(RED_PS.shape + (3,), 255, np.uint8)
difference[RED_PS & ~RED_DONE] = (0, 100, 255)
difference[RED_DONE & ~RED_PS] = (255, 160, 0)
Image.fromarray(difference).save(DESIGN_DIR / '6 changed pixels only.png')
(DESIGN_DIR / '6 stage changes.json').write_text(json.dumps(V5_STAGE_REPORTS, indent=2))

print('=' * 90)
print('CELL 10: CONSERVATIVE FINAL — design', DESIGN)
print('Source: original Photoshop draft from Cell 3.')
print('Broad gap/pocket-fill previews are not selected as the final baseline.')
print('Saved:', bmp)
print('      ', png)
print('Native palette/dimensions:', 'OK' if pure_ok else 'CHECK')
print('Changed pixels:', V5_LAST_FINISHING_REPORT['changed_pixels'], '/', W * H)
print('Red added:', V5_LAST_FINISHING_REPORT['red_added'],
      '| red removed:', V5_LAST_FINISHING_REPORT['red_removed'])
print('Automatic edit budget:', f'{100 * V5_MAX_CHANGED_FRACTION:.2f}%',
      '(this is not a manual-work estimate)')
print('Learned proposal:', V5_LAST_FINISHING_REPORT['learned_proposal'])
print('Detected corners:', V5_LAST_FINISHING_REPORT['audit'])
if V5_LAST_FINISHING_REPORT['changed_pixels'] == 0:
    print('No feasible verified edits accepted: original Photoshop pixels retained for review.')
print('Status:', V5_LAST_EXPORT_REPORT['status'])
print('Stage-by-stage changes relative to Photoshop:')
for row in V5_STAGE_REPORTS.values():
    print(' ', row['stage'], ':', row['changed_pixels_vs_photoshop'], 'pixels changed,',
          row['red_added'], 'red added,', row['red_removed'], 'red removed')

# Designer reference is comparison-only. These scores NEVER choose the output.
if REF is not None:
    comparisons = {'Photoshop': V5_CORE.native_similarity(REF, RED_PS),
                   'Conservative final': V5_CORE.native_similarity(REF, RED_DONE)}
    for label, values in comparisons.items():
        print(label, '— exact agreement:', f"{values['pixel_agreement_pct']:.2f}%",
              '| red IoU:', f"{values['red_iou_pct']:.2f}%",
              '| different pixels:', values['different_pixels'])
    (DESIGN_DIR / '6 native comparisons.json').write_text(json.dumps(comparisons, indent=2))

views = [('Photoshop', RED_PS), ('Conservative FINAL', RED_DONE)]
if REF is not None:
    views.append(('designer — comparison only', REF))
for box in [(0, 0, W, H), (100, 80, 133, 106)]:
    x0 = min(max(0, box[0]), W - 1)
    y0 = min(max(0, box[1]), H - 1)
    bw, bh = min(box[2], W - x0), min(box[3], H - y0)
    fig, axes = plt.subplots(1, len(views), figsize=(6 * len(views), max(2, 6 * bh / bw)), squeeze=False)
    for ax, (label, image) in zip(axes.ravel(), views):
        ax.imshow(rgb_of(image)[y0:y0+bh, x0:x0+bw], interpolation='nearest')
        ax.set_title(f'{label} x{x0}-{x0+bw} y{y0}-{y0+bh}')
        ax.axis('off')
    plt.tight_layout()
    plt.show()
print('The result is a REVIEW DRAFT. Unresolved contours and semantic details still need designer review.')
