# Testing Guide for Website Enhancements

## Quick Test Checklist

### 1. Dark Mode Toggle
**Test Steps:**
1. Open the website in a browser
2. Look for the moon icon (🌙) in the top navigation bar
3. Click the icon
4. ✅ Page should switch to dark theme
5. ✅ Icon should change to sun (☀️)
6. Click again to switch back
7. Refresh the page - theme should persist

**Expected Result:**
- Smooth transition between light and dark themes
- All text remains readable
- Cards and sections adapt to dark background
- Preference saved in browser localStorage

---

### 2. Research Metrics Animation
**Test Steps:**
1. Open the website
2. Scroll down to the "Research Impact" section (purple gradient background)
3. Observe the metrics

**Expected Result:**
- ✅ Numbers animate from 0 to target value
- ✅ Animation triggers when section comes into view
- ✅ Smooth easing effect
- Shows: 10+ Publications, 25+ Citations, 15+ Projects, 8+ Achievements

---

### 3. Publication Filters
**Test Steps:**
1. Navigate to Publications section
2. Look for filter buttons at the top
3. Click "2025" filter
4. ✅ Only 2025 publications should show
5. Click "NLP" filter
6. ✅ Only NLP-related publications should show
7. Click "All Publications"
8. ✅ All publications should reappear

**Expected Result:**
- Smooth fade transitions
- Active button highlighted in blue
- Sticky filter bar when scrolling

---

### 4. Blog Section
**Test Steps:**
1. Scroll to the "Research Blog" section
2. Hover over blog cards
3. ✅ Cards should lift on hover
4. ✅ Images should zoom slightly
5. Check reading time and dates are visible

**Expected Result:**
- 3 blog posts displayed
- Category badges visible
- "View All Posts" button at bottom
- Responsive layout on mobile

---

### 5. Interactive Timeline
**Test Steps:**
1. Navigate to "Career Journey" section
2. Scroll through the timeline
3. Hover over timeline cards
4. ✅ Cards should slide right on hover

**Expected Result:**
- Years displayed prominently (2025, 2024, 2023, etc.)
- Color-coded icons for different event types
- Events listed chronologically
- Vertical line connecting all events

---

### 6. Mobile Responsiveness
**Test Steps:**
1. Open browser DevTools (F12)
2. Toggle device toolbar (Ctrl+Shift+M)
3. Test on different screen sizes:
   - iPhone SE (375px)
   - iPhone 12 Pro (390px)
   - iPad (768px)
   - Desktop (1920px)

**Expected Result:**
- Navigation menu collapses on mobile
- Cards stack vertically
- Images scale properly
- All text remains readable
- No horizontal scroll

---

### 7. PWA Installation
**Test Steps:**
1. Open in Chrome/Edge browser
2. Look for install prompt or menu option
3. Click install
4. ✅ App should install to home screen/desktop
5. Open installed app
6. ✅ Should work offline (after first load)

**For Manual Testing:**
1. Chrome: Menu > Install Koshik Debanath
2. Edge: Menu > Apps > Install this site as an app

**Expected Result:**
- Installs as standalone app
- Works offline
- App icon appears on home screen/desktop

---

### 8. SEO Meta Tags
**Test Steps:**
1. View page source (Ctrl+U)
2. Look for:
   - `<meta property="og:title">`
   - `<meta property="twitter:card">`
   - `<script type="application/ld+json">`
3. Test social sharing:
   - Share URL on Facebook
   - Share URL on Twitter/X
   - Check preview cards

**Tools to Verify:**
- Facebook Debugger: https://developers.facebook.com/tools/debug/
- Twitter Card Validator: https://cards-dev.twitter.com/validator
- Google Rich Results Test: https://search.google.com/test/rich-results

**Expected Result:**
- Rich previews with image and description
- Structured data passes validation
- Proper title and description on social platforms

---

## Browser Testing Matrix

| Feature | Chrome | Firefox | Safari | Edge | Mobile |
|---------|--------|---------|--------|------|--------|
| Dark Mode | ✅ | ✅ | ✅ | ✅ | ✅ |
| Metrics Animation | ✅ | ✅ | ✅ | ✅ | ✅ |
| Filters | ✅ | ✅ | ✅ | ✅ | ✅ |
| Blog Cards | ✅ | ✅ | ✅ | ✅ | ✅ |
| Timeline | ✅ | ✅ | ✅ | ✅ | ✅ |
| PWA Install | ✅ | ⚠️ | ⚠️ | ✅ | ✅ |
| Lazy Loading | ✅ | ✅ | ✅ | ✅ | ✅ |
| Smooth Scroll | ✅ | ✅ | ✅ | ✅ | ✅ |

⚠️ = Limited support but still functional

---

## Performance Testing

### Lighthouse Audit
1. Open DevTools > Lighthouse
2. Select:
   - ✅ Performance
   - ✅ Accessibility
   - ✅ Best Practices
   - ✅ SEO
   - ✅ Progressive Web App
3. Run audit

**Expected Scores:**
- Performance: 90+
- Accessibility: 90+
- Best Practices: 90+
- SEO: 95+
- PWA: Passes all checks

### Page Speed Testing
**Tools:**
- Google PageSpeed Insights: https://pagespeed.web.dev/
- GTmetrix: https://gtmetrix.com/
- WebPageTest: https://www.webpagetest.org/

**Metrics to Check:**
- First Contentful Paint (FCP): < 1.8s
- Largest Contentful Paint (LCP): < 2.5s
- Total Blocking Time (TBT): < 200ms
- Cumulative Layout Shift (CLS): < 0.1

---

## Accessibility Testing

### Keyboard Navigation
1. Use Tab key to navigate
2. Press Enter to activate links/buttons
3. Use arrow keys in menus

**Expected:**
- ✅ All interactive elements reachable
- ✅ Visible focus indicators
- ✅ Logical tab order

### Screen Reader
**Test with:**
- NVDA (Windows)
- JAWS (Windows)
- VoiceOver (Mac/iOS)

**Verify:**
- All images have alt text
- Headings properly structured
- Links descriptive
- Form labels present

### Color Contrast
**Tools:**
- WAVE: https://wave.webaim.org/
- axe DevTools browser extension

**Requirements:**
- Text contrast ratio: 4.5:1 minimum
- Large text: 3:1 minimum
- Works in dark mode too

---

## Common Issues & Solutions

### Issue: Dark mode not persisting
**Solution:** Check browser localStorage is enabled

### Issue: Metrics not animating
**Solution:** Ensure JavaScript is enabled, check console for errors

### Issue: Filters not working
**Solution:** Make sure publication cards have `data-year` and `data-category` attributes

### Issue: PWA not installing
**Solution:** 
- Ensure HTTPS connection
- Check manifest.json is accessible
- Verify service worker registers successfully

### Issue: Images not lazy loading
**Solution:** Check browser supports `loading="lazy"` attribute

---

## Developer Console Checks

### No Errors
```javascript
// Open Console (F12)
// Should see:
"ServiceWorker registered successfully"
// Should NOT see:
// - Any red errors
// - 404 errors for resources
// - JavaScript errors
```

### Network Tab
- All resources loading successfully (200 status)
- Service worker intercepting requests after first load
- Images loading on scroll (lazy loading)

### Application Tab
- Service Worker: Active and running
- Manifest: Loaded correctly
- Storage > Local Storage: `theme` key present
- Cache Storage: Files cached

---

## Mobile-Specific Tests

### Touch Interactions
1. Tap navigation menu
2. Swipe to scroll
3. Tap filter buttons
4. Zoom in/out

**Expected:**
- No accidental clicks
- Buttons large enough (44x44px minimum)
- No horizontal scroll
- Pinch-zoom works

### Orientation
1. Rotate device
2. Switch between portrait/landscape

**Expected:**
- Layout adapts smoothly
- No content cut off
- Navigation remains accessible

---

## Final Verification

### Before Publishing
- [ ] All links work (no 404s)
- [ ] Images load correctly
- [ ] Forms submit properly
- [ ] No console errors
- [ ] Dark mode works
- [ ] Metrics animate
- [ ] Filters function
- [ ] Blog displays correctly
- [ ] Timeline shows all events
- [ ] PWA installs successfully
- [ ] Mobile responsive
- [ ] SEO meta tags present

### After Publishing
- [ ] Test on actual mobile devices
- [ ] Verify social media sharing
- [ ] Check search engine indexing
- [ ] Monitor error logs
- [ ] Gather user feedback

---

## Automated Testing (Optional)

### Jest Tests
```javascript
// Test dark mode toggle
test('theme toggle switches between light and dark', () => {
  // Test implementation
});

// Test publication filters
test('filters show correct publications', () => {
  // Test implementation
});
```

### Cypress E2E Tests
```javascript
// Test user flow
describe('User navigates website', () => {
  it('can toggle dark mode', () => {
    cy.visit('/')
    cy.get('#theme-toggle').click()
    cy.get('html').should('have.attr', 'data-theme', 'dark')
  });
});
```

---

## Performance Monitoring

### Tools
- Google Analytics
- Google Search Console
- Hotjar (heatmaps)
- Sentry (error tracking)

### Metrics to Track
- Page views
- Bounce rate
- Time on page
- PWA installs
- Dark mode usage
- Filter interactions

---

## Success Criteria

✅ **All features work as expected**
✅ **No console errors**
✅ **Lighthouse scores 90+**
✅ **Mobile responsive**
✅ **Fast loading times**
✅ **Accessible to all users**
✅ **SEO optimized**
✅ **PWA installable**

---

**Last Updated:** December 20, 2025
**Version:** 2.0.0
