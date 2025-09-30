# CLAUDE.md - SpaceML Project Context

## 🎯 **Project Mission**
Transform 20-year-old applied math background into space industry AI/ML engineering expertise for roles at **SpaceX, Fleet Space, Gilmour Space, or NASA JPL**. Building production-ready SpaceML library while mastering mathematical foundations.

**Timeline:** 18-24 months intensive study  
**Approach:** MIT-level mathematical rigor + production engineering + real space applications

---

## 🛰️ **SpaceML Library Architecture**

```
spaceml/
├── navigation/      # Coordinate transforms (ECI/ECEF/Body-Fixed) - IN PROGRESS
├── orbital/         # Orbital mechanics & astrodynamics
├── vision/          # Computer vision for space imagery
├── telemetry/       # Spacecraft health monitoring
├── control/         # Spacecraft control & RL
└── data/            # Space data ingestion & processing
```

### **Current Focus: Navigation Module**
- ✅ Vector3D class (immutable, with reference frames)
- ✅ ReferenceFrame enum (ECI, ECEF, BODY)
- ✅ ECI→ECEF transformation (Earth rotation calculations)
- ✅ OrbitState class (position + velocity pairs)
- 🔄 Coordinate transformation registry system
- 🔄 LVLH orbital frame transformations

---

## 📚 **Learning System Structure**

### **Parallel Conversation Channels:**
- **Tensor Temple** 🏛️ - Pure mathematical theory (linear algebra, SVD, eigenvalues)
- **SpaceML Build Lab** ⚡ (this chat) - Production code implementation
- **Pythonic Space Engineering** 🐍 - Python best practices & conventions
- **Mission Control DevOps** 🎛️ - Git, Docker, CI/CD, GCP deployment

### **Current Module Progress:**
**Module 1.1: Linear Algebra & Tensor Mathematics** (~90% complete)
- Vector spaces & coordinate transformations ✅
- Rotation matrices & composition ✅
- Eigenvalue analysis for satellite constellation optimization 🔄
- SVD applications (next focus)

---

## 🎓 **Development Philosophy**

### **3-Layer Framework for Learning:**
1. **Interface Layer** - Clean, professional code interfaces
2. **Orchestration Layer** - System integration & workflows
3. **Operation Layer** - Understanding underlying mechanics

### **Key Principles:**
- **Theory → Practice** - Every math concept applied to space problems
- **Production Focus** - All code deployment-ready
- **Composition over Inheritance** - Clean, extensible architecture
- **Test-alongside Development** - Build, test manually, then write unit tests

---

## 🔧 **Development Environment**

**Tools:**
- WSL2 + Conda + VSCode
- Python 3.11 with scientific stack (NumPy, Pandas, Matplotlib, AstroPy)
- Git with SSH authentication
- Claude Code for seamless development workflow

**Repository:**
- GitHub: `spaceml` (private)
- Branch strategy: feature branches → main
- Commit style: Conventional commits (`feat:`, `fix:`, `docs:`)

---

## 🚀 **Current Sprint: Satellite Constellation Optimization**

### **Active Work (Tensor Temple):**
Building multi-constraint optimization for satellite placement:
- **Efficiency Matrix**: Rewards satellite spreading (1 - overlap)
- **Redundancy Matrix**: Rewards clustering for backup (overlap²)
- **Communication Matrix**: Exponential decay model for inter-satellite links
- **3D Spherical Geometry**: Great circle distances, lat/lon → Cartesian conversion
- **Eigenvalue Optimization**: Finding optimal satellite configurations

### **Active Work (Build Lab - This Chat):**
- Converting lat/lon satellite positions to 3D Cartesian coordinates
- Visualizing satellite constellations in 3D space
- Building matrix formatting tools for debugging constraint matrices
- Preparing to implement complete eigenvalue optimization

---

## 📊 **Key Reference Documents in Project**

1. **claude-project-config.md** - Session templates, progress tracking, project structure
2. **space-ai-learning-system.tsx** - Complete 9-module curriculum with datasets/APIs
3. **complete-curriculum-summary.md** - Full learning roadmap with verified resource links
4. **3-Layer Framework References:**
   - Python Language Reference
   - NumPy Complete Reference  
   - Pandas Complete Reference
   - Matplotlib Complete Reference

---

## 🎯 **Immediate Next Steps**

1. Complete lat/lon → Cartesian conversion for satellite visualization
2. Visualize 3D satellite constellation with Earth sphere
3. Run eigenvalue optimization on multi-constraint system
4. Test different constraint weightings (efficiency vs redundancy vs communication)
5. Integrate constellation optimizer into SpaceML library
6. Build Streamlit interactive constellation designer

---

## 💡 **Context for Claude**

**When continuing conversations:**
- Check current module progress in claude-project-config.md
- Reference 3-layer framework for explanations
- Connect implementations to mathematical theory from Tensor Temple
- Maintain production code quality (type hints, docstrings, error handling)
- Use composition patterns, avoid inheritance unless necessary
- Explain conventions as we go (student learning production development)

**Communication Style:**
- Concise, step-by-step explanations
- Pause between steps to verify understanding
- Provide recommendations but let student implement
- Balance encouragement with MIT-level rigor
- Connect everything to real space industry applications

---

**Last Updated:** Session focused on 3D visualization of satellite constellations and matrix formatting for multi-constraint optimization analysis.

**Energy Level:** Variable - adjust complexity accordingly  
**Current Momentum:** Strong - transitioning theory to production code

🚀 *Building production space software, one elegant algorithm at a time.*