//
//  DrawingView.m
//  FingerTrackDemo
//
//  Created by zjcneil on 2019/2/14.
//  Copyright © 2019 zjcneil. All rights reserved.
//

#import "DrawingView.h"

static int maxDataCount = 28;

@interface DrawingView ()

@property (nonatomic, strong) NSMutableArray *data;

@end

@implementation DrawingView

- (instancetype) initWithCoder:(NSCoder *)aDecoder {
    if (self = [super initWithCoder:aDecoder]) {
        _data = @[].mutableCopy;
    }
    return self;
}

- (UIBezierPath *) straightPath:(NSArray *)points {
    UIBezierPath *path = [UIBezierPath bezierPath];
    if (points.count == 0) {
        return path;
    }
    CGPoint p1 = [points[0] CGPointValue];
    [path moveToPoint:p1];
    for (int i=1; i<points.count; i++) {
        CGPoint the_point = [points[i] CGPointValue];
        [path addLineToPoint:the_point];
    }
    return path;
}

- (CGFloat) getAlpha:(int) index {
    return index * 1.0 / maxDataCount;
}

- (void) drawRect:(CGRect)rect {
    NSTimeInterval time1 = [[NSDate date] timeIntervalSince1970];
    
    if (self.data.count == 0) {
        return;
    }
    if (self.data.count == 1) {
        CGPoint p = [self.data[0] CGPointValue];
        [self drawPointAt:p withColor:UIColor.redColor andRadius:3];
        return;
    }
    
    for (int i=0; i<self.data.count-1; i++) {
        
        CGPoint p1 = [self.data[i] CGPointValue];
        CGPoint p2 = [self.data[i+1] CGPointValue];
        
        [self drawPointAt:p1 withColor:[UIColor colorWithRed:1.0 green:.2 blue:.2 alpha:[self getAlpha:i]] andRadius:3];
        [self drawPointAt:p2 withColor:[UIColor colorWithRed:1.0 green:.2 blue:.2 alpha:[self getAlpha:i+1]] andRadius:3];
        
        UIBezierPath *path = [self straightPath:@[@(p1), @(p2)]];
        [[UIColor colorWithRed:0 green:.8 blue:.3 alpha:[self getAlpha:i+1]] setStroke];
        path.lineWidth = 2.0;
        [path stroke];
    }
}

- (void)append:(CGPoint)p {
    [self.data addObject:@(p)];
    if (self.data.count > maxDataCount) {
        [self.data removeObjectAtIndex:0];
    }
    [self setNeedsDisplay];
}

- (void) drawPointAt:(CGPoint)point withColor:(UIColor *)color andRadius:(CGFloat) radius{
    UIBezierPath *path =[UIBezierPath bezierPathWithOvalInRect:CGRectMake(point.x - radius, point.y - radius, radius*2, radius*2)];
    [color setFill];
    [path fill];
}

@end
