import { t } from "./i18n";
const SELECTED_KEY = "luban.home-background";
const BACKGROUNDS = [
  { id: "free-memories", name: t("时光记忆"), image: "/assets/brand/home-free-memories.webp", min: 0 },
  { id: "free-stickers", name: t("零号贴纸"), image: "/assets/brand/home-free-stickers.webp", min: 0 },
  { id: "welcome", name: t("初次见面"), image: "/assets/brand/home-scene.webp", min: 0 },
  { id: "free-summit", name: t("云海欢呼"), image: "/assets/brand/home-free-summit.webp", min: 0 },
  { id: "free-nap", name: t("云端小憩"), image: "/assets/brand/home-free-nap.webp", min: 0 },
  { id: "free-fishing", name: t("云海垂钓"), image: "/assets/brand/home-free-fishing.webp", min: 0 },
  { id: "free-marbles", name: t("老街弹珠"), image: "/assets/brand/home-free-marbles.webp", min: 0 },
  { id: "free-cards", name: t("屋顶牌局"), image: "/assets/brand/home-free-cards.webp", min: 0 },
  { id: "free-warehouse", name: t("零号仓库"), image: "/assets/brand/home-free-warehouse.webp", min: 0 },
  { id: "free-faces", name: t("表情派对"), image: "/assets/brand/home-free-faces.webp", min: 0 },
  { id: "study", name: t("专心学习"), image: "/assets/brand/home-study.webp", min: 1 },
  { id: "relax", name: t("惬意时光"), image: "/assets/brand/home-relax.webp", min: 1 },
  { id: "inspiration", name: t("灵感时刻"), image: "/assets/brand/home-inspiration.webp", min: 5 },
  { id: "sunset", name: t("落日相伴"), image: "/assets/brand/home-sunset.webp", min: 5 },
  { id: "music", name: t("午后弦音"), image: "/assets/brand/home-music.webp", min: 10 },
  { id: "game", name: t("闲暇时光"), image: "/assets/brand/home-game.webp", min: 10 },
  { id: "milestone-ten", name: t("十次进化"), image: "/assets/brand/home-milestone-ten.webp", min: 10 },
  { id: "night", name: t("星夜阅读"), image: "/assets/brand/home-night.webp", min: 20 },
  { id: "ideas", name: t("晴日灵感"), image: "/assets/brand/home-ideas.webp", min: 20 },
  { id: "castle", name: t("造梦工坊"), image: "/assets/brand/home-castle.webp", min: 30 },
  { id: "painting", name: t("绘梦时光"), image: "/assets/brand/home-painting.webp", min: 30 },
  { id: "camp", name: t("山野露营"), image: "/assets/brand/home-camp.webp", min: 50 },
  { id: "team", name: t("协作之森"), image: "/assets/brand/home-team.webp", min: 50 },
  { id: "journey", name: t("筑梦远行"), image: "/assets/brand/home-journey.webp", min: 70 },
  { id: "overlook", name: t("云端落日"), image: "/assets/brand/home-overlook.webp", min: 70 },
  { id: "valley", name: t("暮色山谷"), image: "/assets/brand/home-valley.webp", min: 100 },
  { id: "hiking", name: t("逐光旅途"), image: "/assets/brand/home-hiking.webp", min: 100 },
  { id: "milestone-hundred", name: t("百次进化"), image: "/assets/brand/home-milestone-hundred.webp", min: 100 },
  { id: "pixel-sunset", name: t("像素晚霞"), image: "/assets/brand/home-pixel-sunset.webp", min: 140 },
  { id: "pixel-music", name: t("像素弦音"), image: "/assets/brand/home-pixel-music.webp", min: 140 },
  { id: "pixel-night", name: t("银河营地"), image: "/assets/brand/home-pixel-night.webp", min: 180 },
  { id: "fishing", name: t("湖畔垂钓"), image: "/assets/brand/home-fishing.webp", min: 180 },
  { id: "seaside", name: t("海风小憩"), image: "/assets/brand/home-seaside.webp", min: 180 },
  { id: "map", name: t("晴空寻路"), image: "/assets/brand/home-map.webp", min: 180 },
  { id: "summit", name: t("山巅远眺"), image: "/assets/brand/home-summit.webp", min: 180 },
  { id: "departure", name: t("向新出发"), image: "/assets/brand/home-departure.webp", min: 180 },
  { id: "victory", name: t("落日加油"), image: "/assets/brand/home-victory.webp", min: 180 },
  { id: "flight", name: t("放飞梦想"), image: "/assets/brand/home-flight.webp", min: 180 },
  { id: "coffee", name: t("晨光同行"), image: "/assets/brand/home-coffee.webp", min: 180 },
  { id: "balcony", name: t("晴日相伴"), image: "/assets/brand/home-balcony.webp", min: 180 },
  { id: "workshop", name: t("创意工坊"), image: "/assets/brand/home-workshop.webp", min: 180 },
  { id: "stroll", name: t("轻快出发"), image: "/assets/brand/home-stroll.webp", min: 180 },
  { id: "comfort", name: t("有你相伴"), image: "/assets/brand/home-comfort.webp", min: 180 },
  { id: "milestone-two-hundred", name: t("两百次进化"), image: "/assets/brand/home-milestone-two-hundred.webp", min: 200 },
] as const;

export function initializeBackgrounds() {
  const dialog = document.getElementById("background-dialog") as HTMLDialogElement;
  const grid = document.getElementById("background-grid")!;
  const progress = document.getElementById("background-progress")!;
  const scene = document.getElementById("home-background") as HTMLImageElement;
  const open = document.getElementById("open-backgrounds")!;
  const close = document.getElementById("close-backgrounds")!;
  let count: number | null = null;
  let preferred: string = BACKGROUNDS[0].id;
  try {
    const saved = localStorage.getItem(SELECTED_KEY);
    if (BACKGROUNDS.some(background => background.id === saved)) preferred = saved!;
  } catch { /* Background selection also works without persistent storage. */ }
  const available = (minimum: number) => minimum === 0 || (count !== null && count >= minimum);
  const cards = BACKGROUNDS.map(background => {
    const card = document.createElement("button");
    card.type = "button"; card.className = "background-card";
    card.dataset.backgroundId = background.id;
    const preview = document.createElement("div"); preview.className = "background-preview";
    const image = document.createElement("img"); image.src = background.image; image.alt = ""; image.loading = "lazy";
    const lock = document.createElement("span"); lock.className = "background-lock";
    lock.textContent = t`未解锁\n${background.min}次进化`;
    preview.append(image, lock);
    const name = document.createElement("strong"); name.textContent = background.name;
    const status = document.createElement("span"); status.className = "background-status";
    card.append(preview, name, status);
    card.addEventListener("click", () => {
      // Check live evidence as well as disabled state, including programmatic clicks.
      if (!available(background.min)) return;
      preferred = background.id;
      try { localStorage.setItem(SELECTED_KEY, preferred); } catch { /* Apply this session's choice anyway. */ }
      render();
      dialog.close();
    });
    return { background, card, lock, status };
  });
  grid.replaceChildren(...cards.map(item => item.card));
  function render() {
    const selected = BACKGROUNDS.find(background => background.id === preferred && available(background.min)) ?? BACKGROUNDS[0];
    if (scene.getAttribute("src") !== selected.image) scene.setAttribute("src", selected.image);
    const unlocked = BACKGROUNDS.filter(background => available(background.min)).length;
    progress.textContent = t`${count === null ? t("进化次数待确认") : t`已完成 ${count} 次进化`} · 已解锁 ${unlocked} / ${BACKGROUNDS.length} 张背景`;
    for (const { background, card, lock, status } of cards) {
      const locked = !available(background.min), active = selected.id === background.id;
      card.disabled = locked;
      card.classList.toggle("is-locked", locked); card.classList.toggle("is-selected", active);
      card.setAttribute("aria-pressed", String(active)); lock.hidden = !locked;
      status.textContent = locked ? t`进化 ${background.min} 次解锁` : active ? t("已选择") : t("点击使用");
      card.setAttribute("aria-label", `${background.name}，${status.textContent}`);
    }
  }
  open.addEventListener("click", () => { if (!dialog.open) dialog.showModal(); });
  close.addEventListener("click", () => dialog.close());
  render();
  return { update(value: number | null | undefined) {
    const next = typeof value === "number" && Number.isSafeInteger(value) && value >= 0 ? value : null;
    if (next === count) return;
    count = next; render();
  } };
}
