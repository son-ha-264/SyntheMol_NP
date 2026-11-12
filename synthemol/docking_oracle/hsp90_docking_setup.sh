#!/usr/bin/env bash
# =============================================================================
# Hsp90alpha NTD receptor + ligand preparation and validation for AutoDock Vina
# (Vinardo scoring), for a two-receptor generative-model docking study.
#
#   4EGK = radicicol      (natural product)  -> NP-induced pocket conformation
#   6LTI = NVP-AUY922     (synthetic)        -> control pocket conformation
#
# Combines and corrects: prepare_hsp90_vina.sh, fix_ligands_vina.sh,
# fix_ligands_rdkit.sh.
#
# Install:
#   conda create -n dock -c conda-forge python=3.11 pymol-open-source vina
#   conda activate dock
#   pip install meeko rdkit pdb2pqr
#
# Usage:
#   bash hsp90_docking_setup.sh            # stages 1-6
#   bash hsp90_docking_setup.sh validate   # re-run validation only
#   bash hsp90_docking_setup.sh sweep      # exhaustiveness + seed diagnostics
#
# STATUS: the pipeline runs end to end, but the setup is NOT yet validated.
# Stage 6 is the gate. Do not start RL runs until it passes.
# =============================================================================
set -euo pipefail

STAGE="${1:-all}"
OUT="${OUT:-hsp90_vina}"
PH=7.4
PAD=5.0
TRIM_LO=16
TRIM_HI=224
EXH="${EXH:-32}"
SEED="${SEED:-42}"

mkdir -p "$OUT"/{raw,aligned,receptors,ligands,box,validation,sweep}
cd "$OUT"

have() { command -v "$1" >/dev/null 2>&1; }
VINA=$(command -v vina || true)
PQR=$(command -v pdb2pqr30 2>/dev/null || command -v pdb2pqr 2>/dev/null || true)

# -----------------------------------------------------------------------------
stage_fetch() {
  echo "=== 1. Structures ==="
  for ID in 4EGK 6LTI; do
    [ -s "raw/${ID}.pdb" ] || curl -fsSL "https://files.rcsb.org/download/${ID}.pdb" -o "raw/${ID}.pdb"
    echo "  raw/${ID}.pdb"
  done
  # Ligand instance coordinates: bond orders come from the Chemical Component
  # Dictionary, NOT guessed from interatomic distances. Essential for radicicol
  # (epoxide, conjugated macrolactone, vinyl chloride).
  [ -s ligands/RDC_raw.sdf ] || curl -fsSL \
    "https://models.rcsb.org/v1/4egk/ligand?auth_seq_id=301&label_asym_id=B&encoding=sdf" \
    -o ligands/RDC_raw.sdf
  [ -s ligands/2GJ_raw.sdf ] || curl -fsSL \
    "https://models.rcsb.org/v1/6lti/ligand?auth_seq_id=301&label_asym_id=B&encoding=sdf" \
    -o ligands/2GJ_raw.sdf
  for F in ligands/*_raw.sdf; do
    echo "  $(basename $F): $(sed -n '4p' $F | awk '{print $1" atoms, "$2" bonds"}')"
  done
  echo "  (RDC 25 atoms = no H in CCD entry; 2GJ 65 atoms = 34 heavy + 31 deposited H)"
}

# -----------------------------------------------------------------------------
stage_align() {
  echo
  echo "=== 2. Superposition + receptor/ligand extraction ==="
  # cealign TARGET, MOBILE -> 4EGK is the target and does not move.
  # Alternate conformations resolved to altloc A explicitly, not left to a default.
  pymol -cq raw/4EGK.pdb raw/6LTI.pdb ligands/2GJ_raw.sdf -d "
    set retain_order, 1
    set pdb_conect_all, 0
    cealign 4EGK, 6LTI
    matrix_copy 6LTI, 2GJ_raw
    save ligands/2GJ_aligned.sdf, 2GJ_raw

    save aligned/4EGK_aln.pdb, 4EGK
    save aligned/6LTI_aln.pdb, 6LTI

    save receptors/4EGK_dry.pdb, 4EGK and polymer.protein and resi ${TRIM_LO}-${TRIM_HI} and not hydro and (alt ''+A)
    save receptors/6LTI_dry.pdb, 6LTI and polymer.protein and resi ${TRIM_LO}-${TRIM_HI} and not hydro and (alt ''+A)
    print '4EGK residues:', cmd.count_atoms(\"4EGK and polymer.protein and resi ${TRIM_LO}-${TRIM_HI} and name CA and (alt ''+A)\")
    print '6LTI residues:', cmd.count_atoms(\"6LTI and polymer.protein and resi ${TRIM_LO}-${TRIM_HI} and name CA and (alt ''+A)\")

    # box reference: union of BOTH ligand envelopes -> chemotype-neutral
    create boxref, (4EGK and resn RDC and not hydro) or (2GJ_raw and not hydro)
    save box/box_union.pdb, boxref
  "
  echo "  >>> cealign RMSD should be < 2 A; both residue counts should read 209 <<<"
}

# -----------------------------------------------------------------------------
stage_receptor() {
  echo
  echo "=== 3. Protonation (pH ${PH}) + receptor PDBQT + box ==="
  for ID in 4EGK 6LTI; do
    if [ -n "$PQR" ]; then
      $PQR --ff=AMBER --with-ph=$PH --titration-state-method=propka --drop-water \
           "receptors/${ID}_dry.pdb" "receptors/${ID}_dry.pqr" \
           > "receptors/${ID}_pdb2pqr.log" 2>&1
      IN_FLAG="--read_pqr"; IN_FILE="receptors/${ID}_dry.pqr"
      echo "  ${ID}: $(grep -c '^ATOM' receptors/${ID}_dry.pqr) atoms after protonation"
    else
      echo "  ${ID}: pdb2pqr unavailable, Meeko will use residue templates"
      IN_FLAG="--read_pdb"; IN_FILE="receptors/${ID}_dry.pdb"
    fi
    mk_prepare_receptor.py $IN_FLAG "$IN_FILE" \
        -o "receptors/${ID}" -p "receptors/${ID}.pdbqt" \
        -v "box/${ID}_vina_box.txt" -j "receptors/${ID}_receptor.json" \
        --box_enveloping box/box_union.pdb --padding $PAD \
        > "receptors/${ID}_meeko.log" 2>&1 \
      || { echo "  FAILED:"; tail -20 "receptors/${ID}_meeko.log"; exit 1; }
  done
  if diff -q box/4EGK_vina_box.txt box/6LTI_vina_box.txt >/dev/null; then
    echo "  boxes identical:"; sed 's/^/    /' box/4EGK_vina_box.txt
  else
    echo "  ERROR: boxes differ -> the superposition failed. Stop here."; exit 1
  fi
}

# -----------------------------------------------------------------------------
stage_ligand() {
  echo
  echo "=== 4. Ligand preparation (RDKit only - Open Babel writes query bonds) ==="
  python3 -I - <<'PY'
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Geometry import Point3D

def load(p, label, sanitize=True):
    m = Chem.MolFromMolFile(p, removeHs=False, sanitize=sanitize)
    if m is None: raise SystemExit(f"RDKit could not parse {p}")
    print(f"  {label}: {m.GetNumAtoms()} atoms, {rdMolDescriptors.CalcMolFormula(m)}")
    return m

# RDC: 4EGK never moved, so coordinates are already in the shared frame.
rdc = Chem.AddHs(load("ligands/RDC_raw.sdf", "RDC raw"), addCoords=True)
Chem.Kekulize(rdc, clearAromaticFlags=True)
Chem.MolToMolFile(rdc, "ligands/RDC_prep.sdf", kekulize=True)
print(f"  RDC prepared: {rdc.GetNumAtoms()} atoms (expect 42)")

# 2GJ: chemistry from the CCD file, coordinates from the superposed file.
raw = load("ligands/2GJ_raw.sdf", "2GJ raw")
aln = Chem.MolFromMolFile("ligands/2GJ_aligned.sdf", removeHs=False, sanitize=False)
if raw.GetNumAtoms() != aln.GetNumAtoms():
    raise SystemExit("atom count mismatch between raw and aligned 2GJ")
if any(a.GetSymbol() != b.GetSymbol() for a, b in zip(raw.GetAtoms(), aln.GetAtoms())):
    raise SystemExit("atom order changed in PyMOL - cannot transfer coordinates")
src, dst = aln.GetConformer(), raw.GetConformer()
for i in range(raw.GetNumAtoms()):
    p = src.GetAtomPosition(i); dst.SetAtomPosition(i, Point3D(p.x, p.y, p.z))
Chem.Kekulize(raw, clearAromaticFlags=True)
Chem.MolToMolFile(raw, "ligands/2GJ_prep.sdf", kekulize=True)
print(f"  2GJ prepared: {raw.GetNumAtoms()} atoms (expect 65)")
print("  CONVENTION: both ligands neutral. Radicicol is neutral at pH 7.4;")
print("  NVP-AUY922's morpholine (pKa ~6-7) is left neutral as deposited.")
print("  Apply the same convention to generated molecules in MolScore.")
PY

  for L in RDC 2GJ; do
    mk_prepare_ligand.py -i "ligands/${L}_prep.sdf" -o "ligands/${L}.pdbqt" \
      || { echo "  FAILED on $L"; exit 1; }
  done
  echo "  NOTE: Meeko makes macrocycles flexible by default (>= v0.3.0). Radicicol's"
  echo "  ring is broken with dummy atoms during docking, which enlarges the search"
  echo "  space - relevant when choosing exhaustiveness."
}

# -----------------------------------------------------------------------------
dock_one() {  # receptor_id ligand_id exhaustiveness seed outdir
  local RID=$1 L=$2 E=$3 S=$4 D=$5
  $VINA --receptor "receptors/${RID}.pdbqt" --ligand "ligands/${L}.pdbqt" \
        --config "box/${RID}_vina_box.txt" \
        --scoring vinardo --exhaustiveness "$E" --num_modes 9 --seed "$S" \
        --out "${D}/${RID}__${L}.pdbqt" > "${D}/${RID}__${L}.log" 2>&1 \
    || { echo "FAILED"; return 1; }
  awk '/^ *1 /{print $2; exit}' "${D}/${RID}__${L}.log"
}

stage_validate() {
  echo
  echo "=== 5. Validation matrix (Vinardo, exhaustiveness ${EXH}, seed ${SEED}) ==="
  printf "  %-8s %-6s %s\n" "receptor" "ligand" "best (kcal/mol)"
  for RID in 4EGK 6LTI; do
    for L in RDC 2GJ; do
      printf "  %-8s %-6s %s\n" "$RID" "$L" "$(dock_one $RID $L $EXH $SEED validation)"
    done
  done

  echo
  echo "=== 6. GATE: symmetry-aware self-docking RMSD ==="
  # Convert docked PDBQT back to SDF with Meeko so connectivity is restored and
  # macrocycle dummy atoms are removed, then use RDKit's symmetry-aware CalcRMS.
  for PAIR in "4EGK RDC" "6LTI 2GJ"; do
    set -- $PAIR; RID=$1; L=$2
    SRC="validation/${RID}__${L}.pdbqt"
    DST="validation/${RID}__${L}_poses.sdf"
    [ -s "$SRC" ] || { echo "  ${RID} x ${L}: no docked output"; continue; }
    # -s writes the SDF; -j passes the receptor json so the export knows the
    # receptor context. Errors are kept, not discarded.
    ELOG="validation/${RID}__${L}_export.log"
    if ! mk_export.py "$SRC" -j "receptors/${RID}_receptor.json" -s "$DST" > "$ELOG" 2>&1; then
      mk_export.py "$SRC" -s "$DST" > "$ELOG" 2>&1 \
        || { echo "  ${RID} x ${L}: mk_export failed:"; sed 's/^/      /' "$ELOG" | tail -8; continue; }
    fi
    echo "  ${RID} x ${L}: exported $(grep -c '\$\$\$\$' "$DST") pose(s)"
  done
  python3 -I - <<'PY'
import os
from rdkit import Chem
from rdkit.Chem import rdMolAlign
for rid, lid in (("4EGK", "RDC"), ("6LTI", "2GJ")):
    poses_f = f"validation/{rid}__{lid}_poses.sdf"
    ref_f   = f"ligands/{lid}_prep.sdf"
    if not os.path.exists(poses_f):
        print(f"  {rid} x {lid}: no exported poses"); continue
    ref = Chem.MolFromMolFile(ref_f, removeHs=True)
    poses = [m for m in Chem.SDMolSupplier(poses_f, removeHs=True) if m is not None]
    if ref is None or not poses:
        print(f"  {rid} x {lid}: parse failed, inspect in PyMOL"); continue
    try:
        best = min(rdMolAlign.CalcRMS(p, ref) for p in poses)
        top  = rdMolAlign.CalcRMS(poses[0], ref)
        flag = "PASS" if top < 2.0 else ("pose 1 off, but a good pose exists" if best < 2.0 else "FAIL")
        print(f"  {rid} x {lid}: pose1 RMSD = {top:.2f} A, best of {len(poses)} = {best:.2f} A   [{flag}]")
    except Exception as e:
        print(f"  {rid} x {lid}: CalcRMS failed ({e}); compare visually in PyMOL")
PY
  cat <<'NOTE'

  Reading the gate:
    pose1 < 2.0 A for BOTH          -> setup validated, proceed.
    pose 1 off but a good pose found -> sampling problem, not a setup problem:
                                        raise exhaustiveness (run the 'sweep' stage).
    no good pose at all              -> wrong pocket description: revisit waters
                                        (the Hsp90 ATP site has conserved waters
                                        bridging Asp93 and Asn51 to resorcinol OH
                                        groups, removed here) before anything else.
NOTE
}

# -----------------------------------------------------------------------------
stage_sweep() {
  echo
  echo "=== Diagnostics: exhaustiveness and seed ==="
  echo "  Radicicol is a flexible macrocycle; NVP-AUY922 is not. If the macrocycle"
  echo "  needs more sampling, the setup penalises exactly the chemistry the NP"
  echo "  prior generates - which would bias the study against the NP arm."
  echo
  printf "  %-6s %-6s %-6s %s\n" "lig" "exh" "seed" "best"
  for L in RDC 2GJ; do
    for E in 32 64 128; do
      printf "  %-6s %-6s %-6s %s\n" "$L" "$E" "$SEED" \
        "$(dock_one 4EGK $L $E $SEED sweep)"
    done
  done
  echo
  echo "  Seed replicates at exhaustiveness ${EXH}:"
  printf "  %-6s %-6s %s\n" "lig" "seed" "best"
  for L in RDC 2GJ; do
    for S in 1 2 3; do
      printf "  %-6s %-6s %s\n" "$L" "$S" "$(dock_one 4EGK $L $EXH $S sweep)"
    done
  done
  echo
  echo "  The spread across seeds is your detection floor. A difference between the"
  echo "  two priors smaller than it is not measurable."
}

# -----------------------------------------------------------------------------
[ -n "$VINA" ] || { echo "vina not found (conda install -c conda-forge vina)"; exit 1; }
have mk_prepare_receptor.py || { echo "meeko not found (pip install meeko)"; exit 1; }

case "$STAGE" in
  all)      stage_fetch; stage_align; stage_receptor; stage_ligand; stage_validate ;;
  fetch)    stage_fetch ;;
  align)    stage_align ;;
  receptor) stage_receptor ;;
  ligand)   stage_ligand ;;
  validate) stage_validate ;;
  sweep)    stage_sweep ;;
  *)        echo "stages: all | fetch | align | receptor | ligand | validate | sweep"; exit 1 ;;
esac
