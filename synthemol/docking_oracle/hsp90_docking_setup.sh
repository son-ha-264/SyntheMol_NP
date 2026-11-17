#!/usr/bin/env bash
# =============================================================================
# Hsp90alpha NTD receptor + ligand preparation and validation, AutoDock Vina 1.2
# with Vinardo scoring, for a two-receptor generative-model docking study.
#
#   4EGK = radicicol      (natural product)  -> NP-induced pocket conformation
#   6LTI = NVP-AUY922     (synthetic)        -> control pocket conformation
#
# Two receptors because the Hsp90 N-terminal lid rearranges with the bound
# ligand. Docking an NP-biased model only into an NP-bound conformation would
# confound a chemotype preference with induced fit.
#
# Install:
#   conda create -n dock -c conda-forge python=3.12 pymol-open-source vina \
#       meeko rdkit pdb2pqr numpy
#   conda activate dock
# Pin python: on 3.13+ there is no RDKit wheel, pip backtracks and installs an
# ancient Meeko that imports the long-removed rdkit.six. Take meeko and rdkit
# from conda-forge, not pip, for the same reason.
#
# Usage:
#   bash hsp90_docking_setup.sh            # stages 1-6
#   bash hsp90_docking_setup.sh validate   # re-run the gate only
#   bash hsp90_docking_setup.sh sweep      # exhaustiveness and seed diagnostics
# Overrides: OUT=hsp90_vina EXH=64 SEED=42 PAD=5.0
#
# STATUS: VALIDATED at exhaustiveness 64.
#   radicicol  in 4EGK: -4.877 +/- 0.027 kcal/mol, redock 0.26 A, 4/4 seeds pass
#   NVP-AUY922 in 6LTI: -8.022 +/- 0.006 kcal/mol, redock 1.78 A (best pose 0.60)
#
# Exhaustiveness is set by the MACROCYCLE, not the rigid ligand:
#   exh 16: radicicol -3.82, RMSD 6.18 A              -> fails outright
#   exh 32: 1 of 4 seeds misses the pose, spread 1.00 kcal/mol
#   exh 64: 4 of 4 pass, spread 0.061 kcal/mol, converged (128 identical)
# NVP-AUY922 is reproducible to 0.014 kcal/mol at all three, so validating on it
# alone would have hidden the problem.
#
# Cross-docking radicicol into 6LTI stays noisy even at 64 (spread 1.12 kcal/mol):
# the synthetic-induced conformation offers macrocycles no deep well. Do not put
# 6LTI in the training reward; use it to re-score final molecules instead.
# =============================================================================
set -euo pipefail

STAGE="${1:-all}"
OUT="${OUT:-hsp90_vina}"
PH=7.4
PAD="${PAD:-5.0}"
TRIM_LO=16            # common modelled range: 4EGK is 11-224, 6LTI is 16-224
TRIM_HI=224
EXH="${EXH:-64}"
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
  # Ligand instance coordinates: bond orders from the Chemical Component
  # Dictionary, not guessed from interatomic distances. Essential for radicicol
  # (epoxide, conjugated macrolactone, vinyl chloride).
  [ -s ligands/RDC_raw.sdf ] || curl -fsSL \
    "https://models.rcsb.org/v1/4egk/ligand?auth_seq_id=301&label_asym_id=B&encoding=sdf" \
    -o ligands/RDC_raw.sdf
  [ -s ligands/2GJ_raw.sdf ] || curl -fsSL \
    "https://models.rcsb.org/v1/6lti/ligand?auth_seq_id=301&label_asym_id=B&encoding=sdf" \
    -o ligands/2GJ_raw.sdf
  for F in ligands/*_raw.sdf; do
    echo "  $(basename "$F"): $(sed -n '4p' "$F" | awk '{print $1" atoms, "$2" bonds"}')"
  done
  echo "  (RDC 25 = no H in the CCD entry; 2GJ 65 = 34 heavy + 31 deposited H)"
}

# -----------------------------------------------------------------------------
stage_align() {
  echo
  echo "=== 2. Superposition + receptor/ligand extraction ==="
  # cealign TARGET, MOBILE -> 4EGK is the target and does not move, so radicicol
  # needs no transform while 2GJ inherits 6LTI's matrix.
  # Alternate conformations resolved to altloc A explicitly, not left to a
  # default: Ser50 is the only altloc residue within 6 A of either ligand.
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
    create boxref, (4EGK and resn RDC and not hydro) or (2GJ_raw and not hydro)
    save box/box_union.pdb, boxref
  "
  echo "  >>> cealign RMSD should be < 2 A; both residue counts should read 209 <<<"
}

# -----------------------------------------------------------------------------
stage_receptor() {
  echo
  echo "=== 3. Protonation (pH ${PH}) + receptor PDBQT + box ==="
  # Waters are dropped: the entries resolve different numbers (112 vs 192, a
  # resolution artefact), so keeping them would give the two arms water networks
  # of unequal completeness. 6LTI's single Mg has no protein atom within 3 A and
  # is crystallisation salt; 4EGK has none, so it is excluded for parity.
  for ID in 4EGK 6LTI; do
    local IN_FLAG IN_FILE
    if [ -n "$PQR" ]; then
      $PQR --ff=AMBER --with-ph=$PH --titration-state-method=propka --drop-water \
           "receptors/${ID}_dry.pdb" "receptors/${ID}_dry.pqr" \
           > "receptors/${ID}_pdb2pqr.log" 2>&1
      IN_FLAG="--read_pqr"; IN_FILE="receptors/${ID}_dry.pqr"
      echo "  ${ID}: $(grep -c '^ATOM' "receptors/${ID}_dry.pqr") atoms after protonation"
    else
      echo "  ${ID}: pdb2pqr unavailable, Meeko will use residue templates"
      IN_FLAG="--read_pdb"; IN_FILE="receptors/${ID}_dry.pdb"
    fi
    # --box_enveloping sizes the box around BOTH reference ligands, so it
    # favours neither chemotype; -v writes the Vina config.
    mk_prepare_receptor.py $IN_FLAG "$IN_FILE" \
        -o "receptors/${ID}" -p "receptors/${ID}.pdbqt" \
        -v "box/${ID}_vina_box.txt" -j "receptors/${ID}_receptor.json" \
        --box_enveloping box/box_union.pdb --padding "$PAD" \
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
  echo "=== 4. Ligand preparation (RDKit only: Open Babel writes query bonds) ==="
  python3 -I - <<'PY'
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Geometry import Point3D

def load(p, label):
    m = Chem.MolFromMolFile(p, removeHs=False, sanitize=True)
    if m is None: raise SystemExit(f"RDKit could not parse {p}")
    print(f"  {label}: {m.GetNumAtoms()} atoms, {rdMolDescriptors.CalcMolFormula(m)}")
    return m

# RDC: 4EGK never moved, so its coordinates are already in the shared frame.
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
print("  Apply the same convention to generated molecules in the oracle.")
PY
  for L in RDC 2GJ; do
    mk_prepare_ligand.py -i "ligands/${L}_prep.sdf" -o "ligands/${L}.pdbqt" \
      || { echo "  FAILED on $L"; exit 1; }
  done
  echo "  NOTE: Meeko makes macrocycles flexible by default (>= 0.3.0), inserting"
  echo "  G0/CG0 glue pseudo-atoms. Vina 1.2 implements that potential; Uni-Dock"
  echo "  and Vina-GPU do not and fail on these ligands."
}

# -----------------------------------------------------------------------------
dock_one() {  # receptor ligand exhaustiveness seed outdir -> best score
  local RID=$1 L=$2 E=$3 S=$4 D=$5
  "$VINA" --receptor "receptors/${RID}.pdbqt" --ligand "ligands/${L}.pdbqt" \
          --config "box/${RID}_vina_box.txt" --scoring vinardo \
          --exhaustiveness "$E" --num_modes 9 --seed "$S" \
          --out "${D}/${RID}__${L}.pdbqt" > "${D}/${RID}__${L}.log" 2>&1 \
    || { echo "FAILED"; return 1; }
  awk '/^ *1 /{print $2; exit}' "${D}/${RID}__${L}.log"
}

stage_validate() {
  echo
  echo "=== 5. Validation matrix (Vinardo, exhaustiveness ${EXH}, seed ${SEED}) ==="

  local MISSING=0
  for L in RDC 2GJ; do
    [ -s "ligands/${L}.pdbqt" ] || { echo "  missing ligands/${L}.pdbqt"; MISSING=1; }
  done
  for RID in 4EGK 6LTI; do
    [ -s "receptors/${RID}.pdbqt" ] || { echo "  missing receptors/${RID}.pdbqt"; MISSING=1; }
    [ -s "box/${RID}_vina_box.txt" ] || { echo "  missing box/${RID}_vina_box.txt"; MISSING=1; }
  done
  [ "$MISSING" = 0 ] || { echo; echo "  Regenerate: bash $0 ligand  (and/or 'receptor')"; exit 1; }

  # Delete previous poses FIRST: otherwise a failed docking leaves the old
  # *_poses.sdf in place and the gate reports the previous run's RMSD as a PASS.
  rm -f validation/*__*_poses.sdf validation/*__*.pdbqt validation/*_export.log

  local FAILED=0 S
  printf "  %-8s %-6s %s\n" "receptor" "ligand" "best (kcal/mol)"
  for RID in 4EGK 6LTI; do
    for L in RDC 2GJ; do
      S=$(dock_one "$RID" "$L" "$EXH" "$SEED" validation)
      [ "$S" = "FAILED" ] && FAILED=1
      printf "  %-8s %-6s %s\n" "$RID" "$L" "$S"
    done
  done
  if [ "$FAILED" = 1 ]; then
    echo; echo "  A docking FAILED, so the gate is skipped - with no fresh poses"
    echo "  it could only report an earlier run. First failure log:"
    for f in validation/*__*.log; do
      [ -s "$f" ] || continue
      echo "  --- $f ---"; sed 's/^/    /' "$f" | tail -12; break
    done
    exit 1
  fi

  echo
  echo "=== 6. GATE: symmetry-aware self-docking RMSD ==="
  # mk_export restores connectivity from the Meeko metadata in the PDBQT and
  # strips the macrocycle dummy atoms; RDKit's CalcRMS is symmetry-aware.
  # Vina's own "rmsd l.b./u.b." columns are pose-to-pose, NOT pose-to-crystal.
  for PAIR in "4EGK RDC" "6LTI 2GJ"; do
    set -- $PAIR; RID=$1; L=$2
    SRC="validation/${RID}__${L}.pdbqt"; DST="validation/${RID}__${L}_poses.sdf"
    [ -s "$SRC" ] || { echo "  ${RID} x ${L}: no docked output"; continue; }
    ELOG="validation/${RID}__${L}_export.log"
    mk_export.py "$SRC" -j "receptors/${RID}_receptor.json" -s "$DST" > "$ELOG" 2>&1 \
      || mk_export.py "$SRC" -s "$DST" > "$ELOG" 2>&1 \
      || { echo "  ${RID} x ${L}: mk_export failed:"; sed 's/^/      /' "$ELOG" | tail -8; continue; }
    echo "  ${RID} x ${L}: exported $(grep -c '\$\$\$\$' "$DST") pose(s)"
  done
  python3 -I - <<'PY'
import os
from rdkit import Chem
from rdkit.Chem import rdMolAlign
for rid, lid in (("4EGK", "RDC"), ("6LTI", "2GJ")):
    poses_f, ref_f = f"validation/{rid}__{lid}_poses.sdf", f"ligands/{lid}_prep.sdf"
    if not os.path.exists(poses_f):
        print(f"  {rid} x {lid}: no exported poses"); continue
    ref = Chem.MolFromMolFile(ref_f, removeHs=True)
    poses = [m for m in Chem.SDMolSupplier(poses_f, removeHs=True) if m is not None]
    if ref is None or not poses:
        print(f"  {rid} x {lid}: parse failed, inspect in PyMOL"); continue
    try:
        best = min(rdMolAlign.CalcRMS(p, ref) for p in poses)
        top = rdMolAlign.CalcRMS(poses[0], ref)
        flag = "PASS" if top < 2.0 else ("pose 1 off, good pose exists" if best < 2.0 else "FAIL")
        print(f"  {rid} x {lid}: pose1 RMSD = {top:.2f} A, best of {len(poses)} = {best:.2f} A   [{flag}]")
    except Exception as e:
        print(f"  {rid} x {lid}: CalcRMS failed ({e}); compare visually")
PY
  cat <<'NOTE'

  Reading the gate:
    pose1 < 2.0 A for BOTH           -> validated, proceed.
    pose 1 off but a good pose found -> sampling: raise EXH and rerun 'sweep'.
    no good pose at all              -> revisit the pocket description. The
                                        Hsp90 ATP site has conserved waters
                                        bridging Asp93 and Asn51 to resorcinol
                                        OH groups, removed here.
NOTE
}

# -----------------------------------------------------------------------------
stage_sweep() {
  echo
  echo "=== Diagnostics: exhaustiveness and seed ==="
  echo "  Radicicol is a flexible macrocycle, NVP-AUY922 is not. Pick"
  echo "  exhaustiveness from the macrocycle: at 32 it misses its crystal pose"
  echo "  in 1 of 4 seeds while AUY922 is reproducible to 0.014 kcal/mol."
  echo
  printf "  %-6s %-6s %-6s %s\n" "lig" "exh" "seed" "best"
  for L in RDC 2GJ; do
    for E in 32 64 128; do
      printf "  %-6s %-6s %-6s %s\n" "$L" "$E" "$SEED" "$(dock_one 4EGK "$L" "$E" "$SEED" sweep)"
    done
  done
  echo
  echo "  Seed replicates at exhaustiveness ${EXH}:"
  printf "  %-6s %-6s %s\n" "lig" "seed" "best"
  for L in RDC 2GJ; do
    for S in 1 2 3; do
      printf "  %-6s %-6s %s\n" "$L" "$S" "$(dock_one 4EGK "$L" "$EXH" "$S" sweep)"
    done
  done
  echo
  echo "  The spread across seeds is your detection floor; a difference between"
  echo "  the two priors smaller than it is not measurable. Check RMSD too, via"
  echo "  'validate' at each setting: a score can be stable with a wrong pose."
}

# -----------------------------------------------------------------------------
[ -n "$VINA" ] || { echo "vina not found (conda install -c conda-forge vina)"; exit 1; }
have mk_prepare_receptor.py || { echo "meeko not found (conda install -c conda-forge meeko)"; exit 1; }

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