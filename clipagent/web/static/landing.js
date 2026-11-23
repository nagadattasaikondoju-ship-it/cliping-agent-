// Landing page interactions: reveal on scroll, nav condense, parallax, clock.
"use strict";

const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

// 1. Reveal elements as they enter the viewport.
const revealObserver = new IntersectionObserver((entries) => {
  for (const entry of entries) {
    if (entry.isIntersecting) {
      entry.target.classList.add("active");
      revealObserver.unobserve(entry.target);
    }
  }
}, { threshold: 0.1, rootMargin: "0px 0px -50px 0px" });
document.querySelectorAll(".reveal").forEach((el) => revealObserver.observe(el));

// 2-4. Nav, card parallax and hero parallax, batched into one animation frame.
const nav = document.getElementById("main-nav");
const hero = document.getElementById("hero-content-wrapper");
const ups = document.querySelectorAll(".parallax-card-up");
const downs = document.querySelectorAll(".parallax-card-down");
let ticking = false;

function onScroll() {
  const y = window.scrollY;
  nav.classList.toggle("scrolled", y > 50);
  if (!reduceMotion) {
    if (window.innerWidth >= 768) {  // card parallax only where the cards sit side by side
      ups.forEach((el) => el.style.setProperty("--scroll-offset-up", `${y * -0.05}px`));
      downs.forEach((el) => el.style.setProperty("--scroll-offset-down", `${y * 0.05}px`));
    }
    if (y < 1000) {
      hero.style.transform = `translateY(${y * 0.4}px)`;
      hero.style.opacity = Math.max(0, 1 - y / 600);
    }
  }
  ticking = false;
}
window.addEventListener("scroll", () => {
  if (!ticking) { ticking = true; requestAnimationFrame(onScroll); }
}, { passive: true });
onScroll();

// 5. Local clock in the hero meta line.
const clock = document.getElementById("current-time");
function updateTime() {
  const now = new Date();
  clock.dateTime = now.toISOString();
  clock.textContent = now.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
}
updateTime();
setInterval(updateTime, 30000);

document.getElementById("year").textContent = new Date().getFullYear();
