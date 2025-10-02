// ----------------- Dark / Light Mode Toggle -----------------
function toggleTheme() {
    let body = document.body;
    body.classList.toggle("dark");
    body.classList.toggle("light");

    let nav = document.querySelector("nav");
    if (body.classList.contains("dark")) {
        nav.classList.remove("navbar-light");
        nav.classList.add("navbar-dark");
    } else {
        nav.classList.remove("navbar-dark");
        nav.classList.add("navbar-light");
    }
}

// ----------------- Manual Grid Animation -----------------
let animationSteps = [];
let currentStep = 0;
let animationInterval = null;
let delay = 80;

// Clear the grid
function clearGrid() {
    document.querySelectorAll("#sudoku-grid input").forEach(input => input.value = '');
    animationSteps = [];
    currentStep = 0;
    if (animationInterval) clearInterval(animationInterval);
}

// Start solving manually entered Sudoku
async function startSolve() {
    let grid = [];
    let inputs = document.querySelectorAll("#sudoku-grid input");
    for (let i = 0; i < 9; i++) {
        let row = [];
        for (let j = 0; j < 9; j++) {
            let val = inputs[i * 9 + j].value;
            row.push(val === '' ? 0 : parseInt(val));
        }
        grid.push(row);
    }

    let res = await fetch('/solve_manual', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ grid: grid })
    });

    let data = await res.json();
    if (data.error) { alert(data.error); return; }

    animationSteps = [];
    for (let i = 0; i < 9; i++) {
        for (let j = 0; j < 9; j++) {
            if (grid[i][j] === 0) {
                animationSteps.push({ row: i, col: j, value: data.solution[i][j] });
            }
        }
    }

    currentStep = 0;
    delay = parseInt(document.getElementById('speedRange').value);
    if (animationInterval) clearInterval(animationInterval);
    animationInterval = setInterval(stepAnimation, delay);
}

// Animate each step
function stepAnimation() {
    let inputs = document.querySelectorAll("#sudoku-grid input");
    if (currentStep >= animationSteps.length) {
        clearInterval(animationInterval);
        return;
    }

    let step = animationSteps[currentStep];
    let index = step.row * 9 + step.col;
    inputs[index].value = step.value;
    inputs[index].classList.add('highlight');
    setTimeout(() => inputs[index].classList.remove('highlight'), delay);

    currentStep++;
}

// Pause animation
function pauseSolve() {
    if (animationInterval) clearInterval(animationInterval);
}

// Reset grid
function resetGrid() {
    clearGrid();
    document.querySelectorAll("#sudoku-grid input").forEach(input => input.value = '');
}

// Update animation speed dynamically
document.getElementById('speedRange').addEventListener('input', function () {
    delay = parseInt(this.value);
    if (animationInterval) {
        clearInterval(animationInterval);
        animationInterval = setInterval(stepAnimation, delay);
    }
});
