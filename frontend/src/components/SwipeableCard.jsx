import React, { useState, useRef } from 'react';

/**
 * Swipeable Card component with touch gesture support
 * Useful for stock cards, portfolio items, etc.
 */
const SwipeableCard = ({ 
  children, 
  onSwipeLeft, 
  onSwipeRight,
  className = '',
  threshold = 100 
}) => {
  const [touchStart, setTouchStart] = useState(null);
  const [touchEnd, setTouchEnd] = useState(null);
  const [swiping, setSwiping] = useState(false);
  const [translateX, setTranslateX] = useState(0);
  const cardRef = useRef(null);

  const minSwipeDistance = threshold;

  const onTouchStart = (e) => {
    setTouchEnd(null);
    setTouchStart(e.targetTouches[0].clientX);
    setSwiping(true);
  };

  const onTouchMove = (e) => {
    const currentTouch = e.targetTouches[0].clientX;
    setTouchEnd(currentTouch);
    
    if (touchStart) {
      const diff = currentTouch - touchStart;
      // Limit the swipe distance
      const limitedDiff = Math.max(-150, Math.min(150, diff));
      setTranslateX(limitedDiff);
    }
  };

  const onTouchEnd = () => {
    if (!touchStart || !touchEnd) {
      setSwiping(false);
      setTranslateX(0);
      return;
    }

    const distance = touchStart - touchEnd;
    const isLeftSwipe = distance > minSwipeDistance;
    const isRightSwipe = distance < -minSwipeDistance;

    if (isLeftSwipe && onSwipeLeft) {
      onSwipeLeft();
    }
    
    if (isRightSwipe && onSwipeRight) {
      onSwipeRight();
    }

    // Reset position
    setSwiping(false);
    setTranslateX(0);
  };

  const cardStyle = {
    transform: swiping ? `translateX(${translateX}px)` : 'translateX(0)',
    transition: swiping ? 'none' : 'transform 0.3s ease',
  };

  return (
    <div
      ref={cardRef}
      onTouchStart={onTouchStart}
      onTouchMove={onTouchMove}
      onTouchEnd={onTouchEnd}
      style={cardStyle}
      className={`swipeable ${className}`}
    >
      {children}
    </div>
  );
};

export default SwipeableCard;
