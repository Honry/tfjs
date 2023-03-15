/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
import * as tfliteWebAPIClient from '../tflite_web_api_client';
import { BaseTaskLibraryClient } from './common';
/**
 * Default NLClassifier options.
 */
const DEFAULT_NLCLASSIFIER_OPTIONS = {
    inputTensorIndex: 0,
    outputScoreTensorIndex: 0,
    outputLabelTensorIndex: -1,
    inputTensorName: 'INPUT',
    outputScoreTensorName: 'OUTPUT_SCORE',
    outputLabelTensorName: 'OUTPUT_LABEL',
};
/**
 * Client for NLClassifier TFLite Task Library.
 *
 * It is a wrapper around the underlying javascript API to make it more
 * convenient to use. See comments in the corresponding type declaration file in
 * src/types for more info.
 */
export class NLClassifier extends BaseTaskLibraryClient {
    constructor(instance) {
        super(instance);
        this.instance = instance;
    }
    static async create(model, options = DEFAULT_NLCLASSIFIER_OPTIONS) {
        const instance = await tfliteWebAPIClient.tfweb.NLClassifier.create(model, options);
        return new NLClassifier(instance);
    }
    classify(input) {
        return this.instance.classify(input).map(category => {
            return {
                probability: category.score,
                className: category.className,
            };
        });
    }
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoibmxfY2xhc3NpZmllci5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtdGZsaXRlL3NyYy90ZmxpdGVfdGFza19saWJyYXJ5X2NsaWVudC9ubF9jbGFzc2lmaWVyLnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUVILE9BQU8sS0FBSyxrQkFBa0IsTUFBTSwwQkFBMEIsQ0FBQztBQUUvRCxPQUFPLEVBQUMscUJBQXFCLEVBQVEsTUFBTSxVQUFVLENBQUM7QUFvQnREOztHQUVHO0FBQ0gsTUFBTSw0QkFBNEIsR0FBd0I7SUFDeEQsZ0JBQWdCLEVBQUUsQ0FBQztJQUNuQixzQkFBc0IsRUFBRSxDQUFDO0lBQ3pCLHNCQUFzQixFQUFFLENBQUMsQ0FBQztJQUMxQixlQUFlLEVBQUUsT0FBTztJQUN4QixxQkFBcUIsRUFBRSxjQUFjO0lBQ3JDLHFCQUFxQixFQUFFLGNBQWM7Q0FDdEMsQ0FBQztBQUVGOzs7Ozs7R0FNRztBQUNILE1BQU0sT0FBTyxZQUFhLFNBQVEscUJBQXFCO0lBQ3JELFlBQStCLFFBQWlDO1FBQzlELEtBQUssQ0FBQyxRQUFRLENBQUMsQ0FBQztRQURhLGFBQVEsR0FBUixRQUFRLENBQXlCO0lBRWhFLENBQUM7SUFFRCxNQUFNLENBQUMsS0FBSyxDQUFDLE1BQU0sQ0FDZixLQUF5QixFQUN6QixPQUFPLEdBQUcsNEJBQTRCO1FBQ3hDLE1BQU0sUUFBUSxHQUNWLE1BQU0sa0JBQWtCLENBQUMsS0FBSyxDQUFDLFlBQVksQ0FBQyxNQUFNLENBQUMsS0FBSyxFQUFFLE9BQU8sQ0FBQyxDQUFDO1FBQ3ZFLE9BQU8sSUFBSSxZQUFZLENBQUMsUUFBUSxDQUFDLENBQUM7SUFDcEMsQ0FBQztJQUVELFFBQVEsQ0FBQyxLQUFhO1FBQ3BCLE9BQU8sSUFBSSxDQUFDLFFBQVEsQ0FBQyxRQUFRLENBQUMsS0FBSyxDQUFDLENBQUMsR0FBRyxDQUFDLFFBQVEsQ0FBQyxFQUFFO1lBQ2xELE9BQU87Z0JBQ0wsV0FBVyxFQUFFLFFBQVEsQ0FBQyxLQUFLO2dCQUMzQixTQUFTLEVBQUUsUUFBUSxDQUFDLFNBQVM7YUFDOUIsQ0FBQztRQUNKLENBQUMsQ0FBQyxDQUFDO0lBQ0wsQ0FBQztDQUNGIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMjEgR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQgKiBhcyB0ZmxpdGVXZWJBUElDbGllbnQgZnJvbSAnLi4vdGZsaXRlX3dlYl9hcGlfY2xpZW50JztcbmltcG9ydCB7TkxDbGFzc2lmaWVyIGFzIFRhc2tMaWJyYXJ5TkxDbGFzc2lmaWVyfSBmcm9tICcuLi90eXBlcy9ubF9jbGFzc2lmaWVyJztcbmltcG9ydCB7QmFzZVRhc2tMaWJyYXJ5Q2xpZW50LCBDbGFzc30gZnJvbSAnLi9jb21tb24nO1xuXG4vKipcbiAqIE5MQ2xhc3NpZmllciBvcHRpb25zLlxuICovXG5leHBvcnQgZGVjbGFyZSBpbnRlcmZhY2UgTkxDbGFzc2lmaWVyT3B0aW9ucyB7XG4gIC8qKiBJbmRleCBvZiB0aGUgaW5wdXQgdGVuc29yLiAqL1xuICBpbnB1dFRlbnNvckluZGV4OiBudW1iZXI7XG4gIC8qKiBJbmRleCBvZiB0aGUgb3V0cHV0IHNjb3JlIHRlbnNvci4gKi9cbiAgb3V0cHV0U2NvcmVUZW5zb3JJbmRleDogbnVtYmVyO1xuICAvKiogSW5kZXggb2YgdGhlIG91dHB1dCBsYWJlbCB0ZW5zb3IuICovXG4gIG91dHB1dExhYmVsVGVuc29ySW5kZXg6IG51bWJlcjtcbiAgLyoqIE5hbWUgb2YgdGhlIGlucHV0IHRlbnNvci4gKi9cbiAgaW5wdXRUZW5zb3JOYW1lOiBzdHJpbmc7XG4gIC8qKiBOYW1lIG9mIHRoZSBvdXRwdXQgc2NvcmUgdGVuc29yLiAqL1xuICBvdXRwdXRTY29yZVRlbnNvck5hbWU6IHN0cmluZztcbiAgLyoqIE5hbWUgb2YgdGhlIG91dHB1dCBsYWJlbCB0ZW5zb3IuICovXG4gIG91dHB1dExhYmVsVGVuc29yTmFtZTogc3RyaW5nO1xufVxuXG4vKipcbiAqIERlZmF1bHQgTkxDbGFzc2lmaWVyIG9wdGlvbnMuXG4gKi9cbmNvbnN0IERFRkFVTFRfTkxDTEFTU0lGSUVSX09QVElPTlM6IE5MQ2xhc3NpZmllck9wdGlvbnMgPSB7XG4gIGlucHV0VGVuc29ySW5kZXg6IDAsXG4gIG91dHB1dFNjb3JlVGVuc29ySW5kZXg6IDAsXG4gIG91dHB1dExhYmVsVGVuc29ySW5kZXg6IC0xLFxuICBpbnB1dFRlbnNvck5hbWU6ICdJTlBVVCcsXG4gIG91dHB1dFNjb3JlVGVuc29yTmFtZTogJ09VVFBVVF9TQ09SRScsXG4gIG91dHB1dExhYmVsVGVuc29yTmFtZTogJ09VVFBVVF9MQUJFTCcsXG59O1xuXG4vKipcbiAqIENsaWVudCBmb3IgTkxDbGFzc2lmaWVyIFRGTGl0ZSBUYXNrIExpYnJhcnkuXG4gKlxuICogSXQgaXMgYSB3cmFwcGVyIGFyb3VuZCB0aGUgdW5kZXJseWluZyBqYXZhc2NyaXB0IEFQSSB0byBtYWtlIGl0IG1vcmVcbiAqIGNvbnZlbmllbnQgdG8gdXNlLiBTZWUgY29tbWVudHMgaW4gdGhlIGNvcnJlc3BvbmRpbmcgdHlwZSBkZWNsYXJhdGlvbiBmaWxlIGluXG4gKiBzcmMvdHlwZXMgZm9yIG1vcmUgaW5mby5cbiAqL1xuZXhwb3J0IGNsYXNzIE5MQ2xhc3NpZmllciBleHRlbmRzIEJhc2VUYXNrTGlicmFyeUNsaWVudCB7XG4gIGNvbnN0cnVjdG9yKHByb3RlY3RlZCBvdmVycmlkZSBpbnN0YW5jZTogVGFza0xpYnJhcnlOTENsYXNzaWZpZXIpIHtcbiAgICBzdXBlcihpbnN0YW5jZSk7XG4gIH1cblxuICBzdGF0aWMgYXN5bmMgY3JlYXRlKFxuICAgICAgbW9kZWw6IHN0cmluZ3xBcnJheUJ1ZmZlcixcbiAgICAgIG9wdGlvbnMgPSBERUZBVUxUX05MQ0xBU1NJRklFUl9PUFRJT05TKTogUHJvbWlzZTxOTENsYXNzaWZpZXI+IHtcbiAgICBjb25zdCBpbnN0YW5jZSA9XG4gICAgICAgIGF3YWl0IHRmbGl0ZVdlYkFQSUNsaWVudC50ZndlYi5OTENsYXNzaWZpZXIuY3JlYXRlKG1vZGVsLCBvcHRpb25zKTtcbiAgICByZXR1cm4gbmV3IE5MQ2xhc3NpZmllcihpbnN0YW5jZSk7XG4gIH1cblxuICBjbGFzc2lmeShpbnB1dDogc3RyaW5nKTogQ2xhc3NbXXx1bmRlZmluZWQge1xuICAgIHJldHVybiB0aGlzLmluc3RhbmNlLmNsYXNzaWZ5KGlucHV0KS5tYXAoY2F0ZWdvcnkgPT4ge1xuICAgICAgcmV0dXJuIHtcbiAgICAgICAgcHJvYmFiaWxpdHk6IGNhdGVnb3J5LnNjb3JlLFxuICAgICAgICBjbGFzc05hbWU6IGNhdGVnb3J5LmNsYXNzTmFtZSxcbiAgICAgIH07XG4gICAgfSk7XG4gIH1cbn1cbiJdfQ==