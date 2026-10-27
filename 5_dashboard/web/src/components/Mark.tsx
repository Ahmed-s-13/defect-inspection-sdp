export function Mark({ size = 22 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden="true">
      <path d="M7 12V7h5M20 7h5v5M25 20v5h-5M12 25H7v-5" fill="none" stroke="var(--ink)" strokeWidth="2.5" />
      <circle cx="16" cy="16" r="2.5" fill="var(--detected)" />
    </svg>
  );
}
