// ----------------- Utilities & State -----------------
let animationSteps = [];
let currentStep = 0;
let animationInterval = null;
let delay = 80;

function getGridInputs() {
    return document.querySelectorAll("#sudoku-grid input");
}

function clearHighlights() {
    getGridInputs().forEach(inp => inp.classList.remove('highlight'));
}

function disableControls() {
    document.querySelectorAll('button, input, select').forEach(el => el.disabled = true);
    const speed = document.getElementById('speedRange');
    if (speed) speed.disabled = false;
}

function enableControls() {
    document.querySelectorAll('button, input, select').forEach(el => el.disabled = false);
}

function stopAnimationInterval() {
    if (animationInterval) {
        clearInterval(animationInterval);
        animationInterval = null;
    }
}

// ----------------- Theme (SVG icons, transitions, tooltip, persistence) -----------------
function setThemeIconAndText(theme) {
    const iconMoon = document.getElementById('icon-moon');
    const iconSun = document.getElementById('icon-sun');
    const themeText = document.getElementById('themeText');
    const btn = document.getElementById('themeToggle');

    if (!iconMoon || !iconSun || !themeText || !btn) return;

    if (theme === 'dark') {
        iconMoon.style.opacity = '0';
        iconSun.style.opacity = '1';
        themeText.textContent = 'Light Mode';
        btn.classList.add('is-dark');
    } else {
        iconMoon.style.opacity = '1';
        iconSun.style.opacity = '0';
        themeText.textContent = 'Dark Mode';
        btn.classList.remove('is-dark');
    }
}

function applyTheme(theme) {
    const body = document.body;
    const nav = document.querySelector("nav");
    if (!body) return;

    body.classList.remove('dark', 'light');
    if (theme === 'dark') {
        body.classList.add('dark');
        if (nav) { nav.classList.remove('navbar-light'); nav.classList.add('navbar-dark'); }
        localStorage.setItem('sudoku_theme', 'dark');
    } else {
        body.classList.add('light');
        if (nav) { nav.classList.remove('navbar-dark'); nav.classList.add('navbar-light'); }
        localStorage.setItem('sudoku_theme', 'light');
    }
    setThemeIconAndText(theme);
}

function toggleTheme() {
    const body = document.body;
    if (!body) return;
    const isDark = body.classList.contains('dark');
    applyTheme(isDark ? 'light' : 'dark');
}

function syncNavWithTheme() {
    const body = document.body;
    const nav = document.querySelector("nav");
    if (!body || !nav) return;
    if (body.classList.contains("dark")) {
        nav.classList.remove("navbar-light");
        nav.classList.add("navbar-dark");
    } else {
        nav.classList.remove("navbar-dark");
        nav.classList.add("navbar-light");
    }
}

function restoreThemePreference() {
    const saved = localStorage.getItem('sudoku_theme');
    if (saved === 'dark' || saved === 'light') {
        applyTheme(saved);
    } else {
        applyTheme('light');
    }
}

// Initialize bootstrap tooltip for theme button and other elements
function initTooltips() {
    if (!window.bootstrap) return;
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.forEach(function (el) {
        try {
            new bootstrap.Tooltip(el);
        } catch (e) {
            // ignore initialization errors
        }
    });
}

// ----------------- Input validation helpers -----------------
function allowOnlySingleDigit(e) {
    const allowedKeys = ['Backspace', 'Tab', 'ArrowLeft', 'ArrowRight', 'Delete'];
    if (allowedKeys.includes(e.key)) return;
    if (!/^[1-9]$/.test(e.key)) {
        e.preventDefault();
    }
}

function preventPasteNonDigit(e) {
    const text = (e.clipboardData || window.clipboardData).getData('text');
    if (!/^[1-9]$/.test(text.trim())) {
        e.preventDefault();
    }
}

function attachInputGuards() {
    const inputs = getGridInputs();
    if (!inputs) return;
    inputs.forEach(inp => {
        inp.setAttribute('inputmode', 'numeric');
        inp.setAttribute('maxlength', '1');
        inp.addEventListener('keydown', allowOnlySingleDigit);
        inp.addEventListener('paste', preventPasteNonDigit);
    });
}

// ----------------- Grid actions -----------------
function clearGrid() {
    getGridInputs().forEach(input => input.value = '');
    animationSteps = [];
    currentStep = 0;
    stopAnimationInterval();
    clearHighlights();
}

function resetGrid() {
    clearGrid();
    getGridInputs().forEach(input => input.value = '');
}

// ----------------- Animation core -----------------
function stepAnimation() {
    const inputs = getGridInputs();
    if (!inputs || currentStep >= animationSteps.length) {
        stopAnimationInterval();
        enableControls();
        return;
    }

    const step = animationSteps[currentStep];
    const index = step.row * 9 + step.col;
    if (inputs[index]) {
        inputs[index].value = step.value;
        inputs[index].classList.add('highlight');
        window.setTimeout(() => {
            if (inputs[index]) inputs[index].classList.remove('highlight');
        }, Math.max(80, delay));
    }

    currentStep++;
    if (currentStep >= animationSteps.length) {
        stopAnimationInterval();
        enableControls();
    }
}

function pauseSolve() {
    stopAnimationInterval();
    enableControls();
}

// ----------------- Solve (manual) -----------------
async function startSolve() {
    const inputs = getGridInputs();
    if (!inputs || inputs.length !== 81) {
        alert('Sudoku grid not found or incomplete.');
        return;
    }

    let grid = [];
    for (let i = 0; i < 9; i++) {
        let row = [];
        for (let j = 0; j < 9; j++) {
            const val = inputs[i * 9 + j].value.trim();
            row.push(val === '' ? 0 : (Number.isInteger(Number(val)) ? parseInt(val) : 0));
        }
        grid.push(row);
    }

    for (let r = 0; r < 9; r++) {
        for (let c = 0; c < 9; c++) {
            const v = grid[r][c];
            if (typeof v !== 'number' || v < 0 || v > 9) {
                alert('Grid contains invalid values. Use digits 1–9 only.');
                return;
            }
        }
    }

    disableControls();
    clearHighlights();

    try {
        const res = await fetch('/solve_manual', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ grid: grid })
        });

        if (!res.ok) {
            throw new Error(`Server returned ${res.status} ${res.statusText}`);
        }

        const data = await res.json();
        if (!data) throw new Error('No response from server.');
        if (data.error) {
            alert(data.error);
            enableControls();
            return;
        }

        animationSteps = [];
        for (let i = 0; i < 9; i++) {
            for (let j = 0; j < 9; j++) {
                if (grid[i][j] === 0) {
                    animationSteps.push({ row: i, col: j, value: data.solution[i][j] });
                }
            }
        }

        currentStep = 0;
        const speedEl = document.getElementById('speedRange');
        delay = speedEl ? parseInt(speedEl.value || delay, 10) : delay;

        stopAnimationInterval();
        if (animationSteps.length > 0) {
            animationInterval = setInterval(stepAnimation, delay);
        } else {
            enableControls();
        }

    } catch (err) {
        console.error('Error in startSolve:', err);
        alert('Failed to solve the puzzle. See console for details.');
        enableControls();
    }
}

// ----------------- Dynamic speed control with persistence -----------------
function updateSpeedFromRange() {
    const speedEl = document.getElementById('speedRange');
    if (!speedEl) return;
    delay = parseInt(speedEl.value, 10) || delay;
    try { localStorage.setItem('sudoku_speed', String(delay)); } catch(e){}
    if (animationInterval) {
        stopAnimationInterval();
        animationInterval = setInterval(stepAnimation, delay);
    }
}

function restoreSpeedPreference() {
    try {
        const saved = localStorage.getItem('sudoku_speed');
        if (saved) {
            const speedEl = document.getElementById('speedRange');
            if (speedEl) {
                speedEl.value = saved;
                delay = parseInt(saved, 10) || delay;
            }
        }
    } catch (e) {
        // ignore
    }
}

// ----------------- Solved image fade-in -----------------
function animateSolvedImage() {
    const img = document.getElementById('solvedImage');
    if (!img) return;
    img.classList.remove('solved-fade-in');
    if (img.complete) {
        img.classList.add('solved-fade-in');
    } else {
        img.addEventListener('load', () => img.classList.add('solved-fade-in'), { once: true });
    }
}

// ----------------- Keyboard shortcuts -----------------
function onKeydownHandler(e) {
    if (e.key === 't' || e.key === 'T') {
        e.preventDefault();
        toggleTheme();
        return;
    }
    const active = document.activeElement;
    if (e.code === 'Space' && active && active.tagName !== 'INPUT' && active.tagName !== 'TEXTAREA') {
        e.preventDefault();
        if (animationInterval) pauseSolve();
        else {
            if (animationSteps && animationSteps.length && currentStep < animationSteps.length) {
                animationInterval = setInterval(stepAnimation, delay);
                disableControls();
            }
        }
    }
}

// ----------------- Init on DOM ready -----------------
function initSudokuScript() {
    attachInputGuards();

    restoreThemePreference();
    restoreSpeedPreference();

    initTooltips();

    animateSolvedImage();

    const speedRange = document.getElementById('speedRange');
    if (speedRange) {
        delay = parseInt(speedRange.value || delay, 10);
        speedRange.addEventListener('input', updateSpeedFromRange);
    }

    syncNavWithTheme();

    const themeToggleBtn = document.getElementById('themeToggle');
    if (themeToggleBtn) themeToggleBtn.addEventListener('click', toggleTheme);

    const startBtn = document.getElementById('startSolveBtn');
    if (startBtn) startBtn.addEventListener('click', startSolve);

    const pauseBtn = document.getElementById('pauseSolveBtn');
    if (pauseBtn) pauseBtn.addEventListener('click', pauseSolve);

    const clearBtn = document.getElementById('clearGridBtn');
    if (clearBtn) clearBtn.addEventListener('click', clearGrid);

    const resetBtn = document.getElementById('resetGridBtn');
    if (resetBtn) resetBtn.addEventListener('click', resetGrid);

    document.addEventListener('keydown', onKeydownHandler);
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initSudokuScript);
} else {
    initSudokuScript();
}
