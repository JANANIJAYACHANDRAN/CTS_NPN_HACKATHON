const express = require("express");
const sqlite3 = require("sqlite3").verbose();
const bodyParser = require("body-parser");
const path = require("path");

const app = express();
const db = new sqlite3.Database("./users.db");

// Serve static files (HTML, CSS, JS, images) from current folder
app.use(express.static(path.join(__dirname)));

// Parse JSON and form data
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));

// Create table if it doesn't exist
db.run(`CREATE TABLE IF NOT EXISTS logins (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)`);

// ✅ Handle login request
app.post("/login", (req, res) => {
    const { username } = req.body;
    if (username) {
        db.run("INSERT INTO logins (username) VALUES (?)", [username], (err) => {
            if (err) {
                console.error("DB insert error:", err.message);
                res.status(500).send("Database error");
            } else {
                console.log("✅ Saved to DB:", username);
                res.json({ success: true });
            }
        });
    } else {
        res.status(400).send("Missing username");
    }
});

// ✅ Extra route to check saved users in browser
app.get("/show-logins", (req, res) => {
    db.all("SELECT * FROM logins ORDER BY id DESC", [], (err, rows) => {
        if (err) {
            res.status(500).send("Database error");
        } else {
            res.send(`
                <h2>Saved Logins</h2>
                <ul>
                  ${rows.map(r => `<li>${r.id}. ${r.username} (at ${r.timestamp})</li>`).join("")}
                </ul>
                <a href="/index.html">🔙 Back to Login Page</a>
            `);
        }
    });
});

// Start server
app.listen(3000, () => {
    console.log("🚀 Server running on http://localhost:3000");
});
